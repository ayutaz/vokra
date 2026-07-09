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
//! # Scope (M3-10-T09 / T10 + follow-up autoregressive forward)
//!
//! This file:
//! - reads the Mistral text decoder weights out of the GGUF (recognising
//!   both the packaged Voxtral prefix `language_model.model.*` and the plain
//!   Mistral prefix `model.*`);
//! - exposes small, unit-testable Rust primitives ([`rms_norm`],
//!   [`silu_inplace`], [`rope_apply`]) that the block forward composes;
//! - implements the full autoregressive block forward — pre-norm, GQA
//!   self-attention with RoPE + causal mask + per-block K/V cache append,
//!   pre-norm SwiGLU FFN, final RMSNorm and tied-logits head — through the
//!   Compute seam so a GPU backend runs the same GEMM path (see
//!   [`forward_step`]);
//! - the KV cache lives on the caller side (`TextDecoderSession`) — one
//!   `KvCache` with width `n_head_kv * head_dim` per layer.
//!
//! The primitives (RMSNorm / SwiGLU / RoPE) are covered by internal oracle
//! tests; the block forward is covered by shape / determinism smoke tests on
//! synthesized weights. Real-checkpoint parity is deferred to a follow-up
//! ticket (T19+) that requires an upstream Voxtral safetensors dump.

use vokra_core::gguf::GgufFile;
use vokra_core::{KvCache, Result, VokraError};

use super::VoxtralConfig;
use crate::compute::Compute;

/// A `nn.Linear` decoded for direct row-major GEMM (`w_t` is `[in, out]`).
///
/// Mistral decoder projections are always **bias-less** (`bias = None`).
///
/// `in_features` / `out_features` are load-time invariants — kept alongside
/// `w_t` for external validation (e.g. audio adapter follow-up sanity
/// checks); silenced from `dead_code` because the block forward reads the
/// shapes off the config, not this struct.
#[allow(dead_code)]
pub(crate) struct Linear {
    pub(crate) w_t: Vec<f32>,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
}

/// A block's four attention projections. GQA: `q` is `[d, n_head_q*head_dim]`
/// = `[d, d]`; `k` / `v` are `[d, n_head_kv*head_dim]`.
pub(crate) struct GqaAttention {
    pub(crate) q: Linear,
    pub(crate) k: Linear,
    pub(crate) v: Linear,
    pub(crate) o: Linear,
}

/// SwiGLU FFN weights: `w2(silu(w1(x)) * w3(x))`.
pub(crate) struct SwiGluFfn {
    pub(crate) gate: Linear, // = w1
    pub(crate) up: Linear,   // = w3
    pub(crate) down: Linear, // = w2
}

/// One Mistral decoder block.
pub(crate) struct DecoderBlock {
    /// RMSNorm γ vector (no bias — RMSNorm is scale-only, `[hidden_dim]`).
    pub(crate) attn_norm_gamma: Vec<f32>,
    pub(crate) attn: GqaAttention,
    pub(crate) ffn_norm_gamma: Vec<f32>,
    pub(crate) ffn: SwiGluFfn,
}

/// All text-decoder weights (tied logits head → the token embedding IS the
/// LM head).
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

// ---------- autoregressive block forward -----------------------------------

/// Scratch buffers for [`forward_step`] — reused across steps by the caller
/// (see [`crate::voxtral::TextDecoderSession`]). Sized to `[max_t_q * d]` at
/// construction; steps up to `max_t_q` reuse without reallocating.
pub(crate) struct StepScratch {
    /// Residual hidden state `[t, d]`.
    pub(crate) h: Vec<f32>,
    /// Pre-norm buffer `[t, d]`.
    pub(crate) norm: Vec<f32>,
    /// Query projection `[t, d]` (n_head_q × head_dim).
    pub(crate) q_proj: Vec<f32>,
    /// Key projection `[t, kv_hidden]` (n_head_kv × head_dim).
    pub(crate) k_proj: Vec<f32>,
    /// Value projection `[t, kv_hidden]`.
    pub(crate) v_proj: Vec<f32>,
    /// One-head Q slice buffer for RoPE `[t, head_dim]`.
    pub(crate) rope_scratch: Vec<f32>,
    /// Attention scores per head `[t, t_kv]`.
    pub(crate) scores: Vec<f32>,
    /// Softmax'd probs per head `[t, t_kv]`.
    pub(crate) probs: Vec<f32>,
    /// Attention output per head `[t, head_dim]`.
    pub(crate) head_out: Vec<f32>,
    /// Concatenated multi-head attention output `[t, d]`.
    pub(crate) attn_out: Vec<f32>,
    /// Post-`o_proj` output `[t, d]`.
    pub(crate) attn_o: Vec<f32>,
    /// FFN gate `[t, ffn_dim]`.
    pub(crate) ffn_gate: Vec<f32>,
    /// FFN up `[t, ffn_dim]`.
    pub(crate) ffn_up: Vec<f32>,
    /// FFN down `[t, d]`.
    pub(crate) ffn_down: Vec<f32>,
    /// Logits per step `[t, vocab_size]`.
    pub(crate) logits: Vec<f32>,
}

impl StepScratch {
    pub(crate) fn with_reserve(
        max_t_q: usize,
        d: usize,
        kv_hidden: usize,
        head_dim: usize,
        ffn_dim: usize,
        vocab_size: usize,
        max_t_kv: usize,
    ) -> Self {
        Self {
            h: Vec::with_capacity(max_t_q * d),
            norm: Vec::with_capacity(max_t_q * d),
            q_proj: Vec::with_capacity(max_t_q * d),
            k_proj: Vec::with_capacity(max_t_q * kv_hidden),
            v_proj: Vec::with_capacity(max_t_q * kv_hidden),
            rope_scratch: Vec::with_capacity(max_t_q * head_dim),
            scores: Vec::with_capacity(max_t_q * max_t_kv),
            probs: Vec::with_capacity(max_t_q * max_t_kv),
            head_out: Vec::with_capacity(max_t_q * head_dim),
            attn_out: Vec::with_capacity(max_t_q * d),
            attn_o: Vec::with_capacity(max_t_q * d),
            ffn_gate: Vec::with_capacity(max_t_q * ffn_dim),
            ffn_up: Vec::with_capacity(max_t_q * ffn_dim),
            ffn_down: Vec::with_capacity(max_t_q * d),
            logits: Vec::with_capacity(max_t_q * vocab_size),
        }
    }
}

fn resize_zero(v: &mut Vec<f32>, len: usize) {
    v.clear();
    v.resize(len, 0.0);
}

/// Runs one decoder step: forwards `tokens` through every block with the
/// caller-owned `kv_cache`, appending each block's K/V rows and leaving the
/// `[t, vocab_size]` logits in `scratch.logits`.
///
/// `position_offset` is the absolute position of `tokens[0]` in the full
/// decode (0 on the first call, then `cache.positions()` before each
/// subsequent call). RoPE uses this offset; the causal mask uses
/// `t_kv = position_offset + t` (past cache rows count) so a step past the
/// first sees prior positions correctly.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] on token out-of-range, decoder not
///   initialised, or `position_offset + t > config.text.n_ctx`.
pub(crate) fn forward_step(
    compute: &Compute,
    cfg: &VoxtralConfig,
    decoder: &TextDecoder,
    scratch: &mut StepScratch,
    kv_cache: &mut KvCache,
    tokens: &[u32],
    position_offset: usize,
) -> Result<()> {
    let d = cfg.text.hidden_dim;
    let ffn_dim = cfg.text.ffn_dim;
    let vocab = cfg.text.vocab_size;
    let n_head_q = cfg.text.n_head_q;
    let n_head_kv = cfg.text.n_head_kv;
    let n_layer = cfg.text.n_layer;
    let rope_base = cfg.text.rope_base;
    let eps = cfg.text.rms_norm_eps;
    let t = tokens.len();

    if t == 0 {
        return Ok(());
    }
    if d == 0 || vocab == 0 || n_head_q == 0 || n_head_kv == 0 || n_layer == 0 {
        return Err(VokraError::ModelLoad(
            "voxtral text_decoder.forward_step: config carries 0-sentinel — re-convert with a \
             full VoxtralConfig (FR-EX-08 — no silent default)."
                .into(),
        ));
    }
    if position_offset + t > cfg.text.n_ctx {
        return Err(VokraError::InvalidArgument(format!(
            "voxtral text_decoder.forward_step: position {} > n_ctx {}",
            position_offset + t,
            cfg.text.n_ctx
        )));
    }
    if n_head_q % n_head_kv != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "voxtral text_decoder.forward_step: n_head_q ({n_head_q}) must be divisible by n_head_kv ({n_head_kv}) — GQA"
        )));
    }
    let head_dim = d / n_head_q;
    let kv_hidden = n_head_kv * head_dim;
    let n_kv_groups = n_head_q / n_head_kv;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    if decoder.blocks.len() != n_layer {
        return Err(VokraError::ModelLoad(format!(
            "voxtral text_decoder.forward_step: loaded blocks {} != config n_layer {n_layer}",
            decoder.blocks.len()
        )));
    }
    if decoder.token_emb.len() != vocab * d {
        return Err(VokraError::ModelLoad(format!(
            "voxtral text_decoder.forward_step: token_emb len {} != vocab*d {}",
            decoder.token_emb.len(),
            vocab * d
        )));
    }

    // Token embedding lookup into scratch.h.
    resize_zero(&mut scratch.h, t * d);
    for (i, &tok) in tokens.iter().enumerate() {
        let tok = tok as usize;
        if tok >= vocab {
            return Err(VokraError::InvalidArgument(format!(
                "voxtral text_decoder.forward_step: token id {tok} >= vocab {vocab}"
            )));
        }
        let src = &decoder.token_emb[tok * d..tok * d + d];
        let dst = &mut scratch.h[i * d..i * d + d];
        dst.copy_from_slice(src);
    }

    // Pre-size mutable scratch (avoid per-block reallocation).
    resize_zero(&mut scratch.norm, t * d);
    resize_zero(&mut scratch.q_proj, t * d);
    resize_zero(&mut scratch.k_proj, t * kv_hidden);
    resize_zero(&mut scratch.v_proj, t * kv_hidden);
    resize_zero(&mut scratch.rope_scratch, t * head_dim);
    resize_zero(&mut scratch.attn_out, t * d);
    resize_zero(&mut scratch.attn_o, t * d);
    resize_zero(&mut scratch.ffn_gate, t * ffn_dim);
    resize_zero(&mut scratch.ffn_up, t * ffn_dim);
    resize_zero(&mut scratch.ffn_down, t * d);

    for (layer_idx, block) in decoder.blocks.iter().enumerate() {
        // ---------- Pre-norm self-attention ----------
        rms_norm(
            &scratch.h,
            &block.attn_norm_gamma,
            eps,
            t,
            &mut scratch.norm,
        )?;

        // Q = norm @ q.w_t: [t, d] × [d, d] → [t, d]
        compute.gemm_f32(
            t,
            d,
            d,
            &scratch.norm,
            &block.attn.q.w_t,
            None,
            &mut scratch.q_proj,
        )?;
        // K = norm @ k.w_t: [t, d] × [d, kv_hidden] → [t, kv_hidden]
        compute.gemm_f32(
            t,
            kv_hidden,
            d,
            &scratch.norm,
            &block.attn.k.w_t,
            None,
            &mut scratch.k_proj,
        )?;
        // V = norm @ v.w_t: [t, d] × [d, kv_hidden] → [t, kv_hidden]
        compute.gemm_f32(
            t,
            kv_hidden,
            d,
            &scratch.norm,
            &block.attn.v.w_t,
            None,
            &mut scratch.v_proj,
        )?;

        // Apply RoPE per-head to Q and K.
        for h in 0..n_head_q {
            // Extract head h's Q slice into rope_scratch, apply RoPE, write back.
            for i in 0..t {
                let src = &scratch.q_proj[i * d + h * head_dim..i * d + (h + 1) * head_dim];
                scratch.rope_scratch[i * head_dim..(i + 1) * head_dim].copy_from_slice(src);
            }
            rope_apply(
                &mut scratch.rope_scratch[..t * head_dim],
                t,
                head_dim,
                rope_base,
                position_offset,
            )?;
            for i in 0..t {
                let dst = &mut scratch.q_proj[i * d + h * head_dim..i * d + (h + 1) * head_dim];
                dst.copy_from_slice(&scratch.rope_scratch[i * head_dim..(i + 1) * head_dim]);
            }
        }
        for h in 0..n_head_kv {
            for i in 0..t {
                let src = &scratch.k_proj
                    [i * kv_hidden + h * head_dim..i * kv_hidden + (h + 1) * head_dim];
                scratch.rope_scratch[i * head_dim..(i + 1) * head_dim].copy_from_slice(src);
            }
            rope_apply(
                &mut scratch.rope_scratch[..t * head_dim],
                t,
                head_dim,
                rope_base,
                position_offset,
            )?;
            for i in 0..t {
                let dst = &mut scratch.k_proj
                    [i * kv_hidden + h * head_dim..i * kv_hidden + (h + 1) * head_dim];
                dst.copy_from_slice(&scratch.rope_scratch[i * head_dim..(i + 1) * head_dim]);
            }
        }

        // Append K/V to cache.
        kv_cache.append(
            layer_idx,
            &scratch.k_proj[..t * kv_hidden],
            &scratch.v_proj[..t * kv_hidden],
        );
        let t_kv = position_offset + t;
        let k_cache = kv_cache.k(layer_idx);
        let v_cache = kv_cache.v(layer_idx);
        // K/V cache rows for layer_idx: [t_kv, kv_hidden].

        // Attention: for each Q head h_q, use K/V head h_kv = h_q / n_kv_groups.
        // scores[t, t_kv] = Q_h @ K_h.T * scale
        // apply causal mask (row i can attend up to position_offset + i)
        // probs = softmax(scores)
        // attn_head[t, head_dim] = probs @ V_h
        // scatter head output into scratch.attn_out [t, d].
        resize_zero(&mut scratch.scores, t * t_kv);
        resize_zero(&mut scratch.probs, t * t_kv);
        resize_zero(&mut scratch.head_out, t * head_dim);
        for h_q in 0..n_head_q {
            let h_kv = h_q / n_kv_groups;
            // scores[i, j] = Σ_c Q[i, h_q*head_dim + c] * K[j, h_kv*head_dim + c] * scale
            for i in 0..t {
                let q_row = &scratch.q_proj[i * d + h_q * head_dim..i * d + (h_q + 1) * head_dim];
                let row_start = i * t_kv;
                for j in 0..t_kv {
                    let k_row = &k_cache
                        [j * kv_hidden + h_kv * head_dim..j * kv_hidden + (h_kv + 1) * head_dim];
                    let mut s = 0.0f32;
                    for c in 0..head_dim {
                        s += q_row[c] * k_row[c];
                    }
                    scratch.scores[row_start + j] = s * scale;
                }
                // Causal mask: row i's absolute position is position_offset + i,
                // so keys at j > position_offset + i are masked out.
                let cur_pos = position_offset + i;
                for j in (cur_pos + 1)..t_kv {
                    scratch.scores[row_start + j] = f32::NEG_INFINITY;
                }
            }
            // Row-wise softmax.
            compute.softmax_f32(&scratch.scores, &mut scratch.probs, t, t_kv)?;
            // head_out[i, c] = Σ_j probs[i, j] * V[j, h_kv*head_dim + c]
            for i in 0..t {
                let dst = &mut scratch.head_out[i * head_dim..(i + 1) * head_dim];
                for c in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..t_kv {
                        let v_row = &v_cache[j * kv_hidden + h_kv * head_dim
                            ..j * kv_hidden + (h_kv + 1) * head_dim];
                        sum += scratch.probs[i * t_kv + j] * v_row[c];
                    }
                    dst[c] = sum;
                }
                // Scatter into scratch.attn_out at the h_q head slot.
                let out_dst =
                    &mut scratch.attn_out[i * d + h_q * head_dim..i * d + (h_q + 1) * head_dim];
                out_dst.copy_from_slice(dst);
            }
        }

        // O projection: attn_out @ o.w_t: [t, d] × [d, d] → [t, d]
        compute.gemm_f32(
            t,
            d,
            d,
            &scratch.attn_out,
            &block.attn.o.w_t,
            None,
            &mut scratch.attn_o,
        )?;

        // Residual add.
        for i in 0..t * d {
            scratch.h[i] += scratch.attn_o[i];
        }

        // ---------- Pre-norm SwiGLU FFN ----------
        rms_norm(&scratch.h, &block.ffn_norm_gamma, eps, t, &mut scratch.norm)?;
        // gate = norm @ gate.w_t → [t, ffn_dim]
        compute.gemm_f32(
            t,
            ffn_dim,
            d,
            &scratch.norm,
            &block.ffn.gate.w_t,
            None,
            &mut scratch.ffn_gate,
        )?;
        // up = norm @ up.w_t → [t, ffn_dim]
        compute.gemm_f32(
            t,
            ffn_dim,
            d,
            &scratch.norm,
            &block.ffn.up.w_t,
            None,
            &mut scratch.ffn_up,
        )?;
        // silu(gate) * up
        silu_inplace(&mut scratch.ffn_gate);
        hadamard_inplace(&mut scratch.ffn_gate, &scratch.ffn_up)?;
        // down = (silu(gate) * up) @ down.w_t → [t, d]
        compute.gemm_f32(
            t,
            d,
            ffn_dim,
            &scratch.ffn_gate,
            &block.ffn.down.w_t,
            None,
            &mut scratch.ffn_down,
        )?;
        // Residual add.
        for i in 0..t * d {
            scratch.h[i] += scratch.ffn_down[i];
        }
    }
    // Advance the position clock once (after all layer appends).
    kv_cache.advance(t);

    // Final RMSNorm.
    rms_norm(
        &scratch.h,
        &decoder.final_norm_gamma,
        eps,
        t,
        &mut scratch.norm,
    )?;

    // Tied logits head: logits[t, vocab] = norm[t, d] × token_emb.T[d, vocab]
    // token_emb is stored as [vocab, d] (row-major). For row-major GEMM the
    // formulation is: logits = norm × (token_emb).T
    // The gemm_f32 API is `C[m,n] = A[m,k] × B[k,n]`; we want
    //   logits[t, vocab] = norm[t, d] × token_embT[d, vocab]
    // and token_embT[d*vocab] can be gotten by treating token_emb as an
    // [vocab, d] matrix and using gemm with an implicit transpose — but
    // gemm_f32 has no transpose flag. So we compute row-by-row via gemv:
    //   logits[i, v] = Σ_c norm[i, c] * token_emb[v, c]
    //   ⇒ logits_row_i = gemv(m=vocab, k=d, a=token_emb, x=norm_row_i)
    resize_zero(&mut scratch.logits, t * vocab);
    for i in 0..t {
        let x = &scratch.norm[i * d..(i + 1) * d];
        let out = &mut scratch.logits[i * vocab..(i + 1) * vocab];
        compute.gemv_f32(vocab, d, &decoder.token_emb, x, None, out)?;
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

    // ---------- extended oracle tests (M3-10 structural completion) --------

    #[test]
    fn rms_norm_scales_by_gamma_per_channel() {
        // With a non-uniform γ, each column of the output should be scaled
        // exactly by the corresponding γ[c] after the row is normalised.
        // Craft a row whose RMS is a nice number so the effect of γ is
        // isolated from the divisor.
        let d = 4;
        // row [2, 2, 2, 2] has mean(x^2)=4 → RMS=2 → x / RMS = [1, 1, 1, 1].
        let x = vec![2.0f32; d];
        // γ = [10, 20, 30, 40] → out = γ * 1.
        let gamma = vec![10.0f32, 20.0, 30.0, 40.0];
        let mut out = vec![0.0f32; d];
        rms_norm(&x, &gamma, 0.0, 1, &mut out).unwrap();
        for (i, &g) in gamma.iter().enumerate() {
            assert!(
                (out[i] - g).abs() < 1e-4,
                "column {i}: expected {g}, got {}",
                out[i]
            );
        }
    }

    #[test]
    fn rms_norm_epsilon_prevents_divide_by_zero_and_scales_predictably() {
        // Non-zero row with a large ε: the divisor becomes sqrt(mean_sq + ε).
        // For row [2,2,2,2] mean_sq=4, ε=12 → divisor=sqrt(16)=4 → out = x/4 = [0.5,…].
        let x = vec![2.0f32; 4];
        let gamma = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        rms_norm(&x, &gamma, 12.0, 1, &mut out).unwrap();
        for v in &out {
            assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
        }
    }

    #[test]
    fn rms_norm_multirow_processes_each_row_independently() {
        // Two rows with different scales must be normalised to the same RMS.
        let d = 4;
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let gamma = vec![1.0f32; d];
        let mut out = vec![0.0f32; d * 2];
        rms_norm(&x, &gamma, 0.0, 2, &mut out).unwrap();
        for row in 0..2 {
            let slice = &out[row * d..(row + 1) * d];
            let rms = (slice.iter().map(|v| v * v).sum::<f32>() / d as f32).sqrt();
            assert!(
                (rms - 1.0).abs() < 1e-4,
                "row {row}: RMS should be 1, got {rms}"
            );
        }
    }

    #[test]
    fn rms_norm_shape_mismatch_is_error_not_panic() {
        // x/out length disagreeing with rows*d must surface as an error.
        let gamma = vec![1.0f32; 4];
        let x = vec![1.0f32; 3]; // should be 4 for one row
        let mut out = vec![0.0f32; 4];
        assert!(matches!(
            rms_norm(&x, &gamma, 0.0, 1, &mut out),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn silu_derivative_positive_at_origin() {
        // SiLU(0)=0 and SiLU'(0)=0.5. This is a small numerical check that
        // silu_inplace matches the math (numerical derivative via
        // (silu(h) - silu(-h)) / 2h).
        let h = 1e-3f32;
        let mut a = vec![h];
        let mut b = vec![-h];
        silu_inplace(&mut a);
        silu_inplace(&mut b);
        let d = (a[0] - b[0]) / (2.0 * h);
        assert!((d - 0.5).abs() < 1e-2, "silu'(0) ≈ 0.5, got {d}");
    }

    #[test]
    fn silu_asymptotic_saturation() {
        // silu(large positive x) ≈ x; silu(large negative x) ≈ 0.
        let mut pos = vec![50.0f32];
        let mut neg = vec![-50.0f32];
        silu_inplace(&mut pos);
        silu_inplace(&mut neg);
        assert!((pos[0] - 50.0).abs() < 1e-3, "silu(50)≈50, got {}", pos[0]);
        assert!(neg[0].abs() < 1e-10, "silu(-50)≈0, got {}", neg[0]);
    }

    #[test]
    fn swiglu_gate_up_roundtrip_pattern() {
        // SwiGLU is `silu(gate(x)) * up(x)`. Verify the pattern element-wise
        // using pre-computed gate and up projections on a small vector.
        // For x=[1,2,3,4] with an identity gate and up: silu(x)*x should be
        // silu-elementwise times x-elementwise.
        let gate_out = vec![1.0f32, 2.0, 3.0, 4.0];
        let up_out = vec![1.0f32, 2.0, 3.0, 4.0];
        // Apply silu to a copy of gate_out.
        let mut activated = gate_out.clone();
        silu_inplace(&mut activated);
        // Hadamard with up_out.
        let mut swiglu = activated.clone();
        hadamard_inplace(&mut swiglu, &up_out).unwrap();
        // Verify each element: silu(gate[i]) * up[i].
        for (i, ((&g, &u), &s)) in gate_out.iter().zip(&up_out).zip(&swiglu).enumerate() {
            let expected = g * (1.0 / (1.0 + (-g).exp())) * u;
            assert!(
                (s - expected).abs() < 1e-4,
                "swiglu[{i}] expected {expected}, got {s}"
            );
        }
    }

    #[test]
    fn rope_apply_frequency_formula_at_first_pair() {
        // Verify the first frequency pair (j=0) rotates by angle m * θ_0 =
        // m * rope_base^(-2*0/head_dim) = m * 1 = m radians (regardless of
        // rope_base). This is a bedrock property: the θ_0 pair rotates at
        // exactly the position rate.
        let head_dim = 4;
        let rope_base = 10_000.0f32;
        let m = 5.0f32;
        // Row [1, 0, 0, 0]: the (j=0) pair is (x[0]=1, x[2]=0).
        // After RoPE at position m: x[0]=cos(m*1)*1 = cos(m), x[2]=sin(m).
        let mut x = vec![1.0f32, 0.0, 0.0, 0.0];
        rope_apply(&mut x, 1, head_dim, rope_base, m as usize).unwrap();
        assert!(
            (x[0] - m.cos()).abs() < 1e-4,
            "x[0]={}, want cos({m})",
            x[0]
        );
        assert!(
            (x[2] - m.sin()).abs() < 1e-4,
            "x[2]={}, want sin({m})",
            x[2]
        );
    }

    #[test]
    fn rope_apply_second_pair_scales_frequency_with_rope_base() {
        // For the j=1 pair, θ_1 = rope_base^(-2/head_dim).
        // With head_dim=4 and rope_base=10_000, θ_1 = 10_000^(-0.5) = 0.01.
        // Row [0, 1, 0, 0]: the (j=1) pair is (x[1]=1, x[3]=0).
        // After RoPE at position m=1: x[1]=cos(θ_1), x[3]=sin(θ_1).
        let head_dim = 4;
        let rope_base = 10_000.0f32;
        let theta_1 = rope_base.powf(-2.0 / head_dim as f32);
        let mut x = vec![0.0f32, 1.0, 0.0, 0.0];
        rope_apply(&mut x, 1, head_dim, rope_base, 1).unwrap();
        assert!(
            (x[1] - theta_1.cos()).abs() < 1e-5,
            "x[1]={}, want cos({theta_1})",
            x[1]
        );
        assert!(
            (x[3] - theta_1.sin()).abs() < 1e-5,
            "x[3]={}, want sin({theta_1})",
            x[3]
        );
    }

    #[test]
    fn rope_apply_position_offset_advances_angles_by_one_row() {
        // A single row at offset m must equal the m-th row of a run at
        // offset 0 with m+1 rows. This is the incremental-decoding
        // invariant that KV-cache-append depends on.
        let head_dim = 4;
        let rope_base = 10_000.0f32;
        let orig = [1.0f32, 2.0, 3.0, 4.0];
        // Full-range run at offset 0, 5 rows: use row 3.
        let mut full = orig.repeat(5);
        rope_apply(&mut full, 5, head_dim, rope_base, 0).unwrap();
        let row_from_full = &full[3 * head_dim..4 * head_dim];
        // Single-row run at offset 3.
        let mut single = orig.to_vec();
        rope_apply(&mut single, 1, head_dim, rope_base, 3).unwrap();
        for (i, (&a, &b)) in single.iter().zip(row_from_full.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "offset invariance broken at index {i}: single={a}, cached={b}"
            );
        }
    }

    #[test]
    fn rope_apply_length_mismatch_is_error_not_panic() {
        let mut x = vec![1.0f32, 2.0, 3.0]; // 3 elements, seq_len*head_dim=4
        assert!(matches!(
            rope_apply(&mut x, 1, 4, 10_000.0, 0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn gqa_head_split_derivation_from_config() {
        // Voxtral-mini-3B ships (n_head_q=24, n_head_kv=8, hidden_dim=3072)
        // → head_dim=128, n_kv_groups=24/8=3 → each K/V head is broadcast
        // to 3 query heads. Verify the config's head_dim() computation.
        use crate::voxtral::config::TextDecoderConfig;
        let cfg = TextDecoderConfig {
            n_layer: 28,
            n_head_q: 24,
            n_head_kv: 8,
            hidden_dim: 3072,
            ffn_dim: 8192,
            vocab_size: 32_000,
            n_ctx: 32_768,
            rope_base: 1_000_000.0,
            rms_norm_eps: 1e-5,
        };
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(
            cfg.n_head_q % cfg.n_head_kv,
            0,
            "GQA requires n_head_q % n_head_kv == 0"
        );
        // The number of query heads sharing one K/V head:
        assert_eq!(cfg.n_head_q / cfg.n_head_kv, 3);
    }
}
