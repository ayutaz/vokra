//! Whisper text decoder forward pass with a KV cache.
//!
//! Structure (openai/whisper `TextDecoder`, HF `WhisperDecoder`):
//!
//! - token embedding + learned positional embedding (`scale_embedding = false`,
//!   so no `√d` scaling);
//! - `n_text_layer` blocks of: pre-norm **causal** self-attention → pre-norm
//!   **cross**-attention over the encoder output → pre-norm MLP;
//! - a final LayerNorm and a logits projection **tied** to the token embedding
//!   (`logits = h · embed_tokensᵀ`, no bias).
//!
//! # KV cache (model-internal, M0)
//!
//! [`DecoderState`] keeps two kinds of cache:
//!
//! - **cross-attention** K/V, computed **once** from the encoder output and
//!   reused for every decode step;
//! - **self-attention** K/V, appended each step so past tokens are not
//!   recomputed.
//!
//! Promoting this to a first-class session state (FR-EX-02) is M1-04; here it
//! is a private implementation detail that never appears on the graph I/O. A
//! full-sequence call ([`DecoderState::step`] on the whole prefix) and a
//! token-by-token cached call produce identical logits (verified by the parity
//! tests), which is what the search integration (M0-06-T23) relies on.

use vokra_backend_cpu::kernels::gemm_f32;
use vokra_core::{Result, VokraError};

use super::config::WhisperConfig;
use super::encoder::EncoderOutput;
use super::nn::{add_into, attention_from_kv, layer_norm, mlp, project_kv};
use super::weights::DecoderWeights;

/// A decoder run bound to one encoder output, holding the KV caches.
pub struct DecoderState<'a> {
    cfg: &'a WhisperConfig,
    w: &'a DecoderWeights,
    /// Per-layer cross-attention `(k, v)`, each `[n_ctx, d]` (computed once).
    cross_kv: Vec<(Vec<f32>, Vec<f32>)>,
    /// Number of encoder context positions.
    n_ctx: usize,
    /// Per-layer self-attention key cache, growable `[pos, d]`.
    self_k: Vec<Vec<f32>>,
    /// Per-layer self-attention value cache, growable `[pos, d]`.
    self_v: Vec<Vec<f32>>,
    /// Number of tokens already committed to the self-attention cache.
    pos: usize,
}

impl<'a> DecoderState<'a> {
    /// Binds to `encoder` and precomputes the cross-attention K/V for every
    /// layer.
    pub(crate) fn new(
        cfg: &'a WhisperConfig,
        w: &'a DecoderWeights,
        encoder: &EncoderOutput,
    ) -> Result<Self> {
        if encoder.d_model != cfg.d_model {
            return Err(VokraError::InvalidArgument(format!(
                "whisper decoder: encoder d_model {} != config {}",
                encoder.d_model, cfg.d_model
            )));
        }
        let mut cross_kv = Vec::with_capacity(w.layers.len());
        for layer in &w.layers {
            cross_kv.push(project_kv(
                &encoder.hidden,
                encoder.n_ctx,
                &layer.cross_attn,
            )?);
        }
        let n_layer = w.layers.len();
        Ok(Self {
            cfg,
            w,
            cross_kv,
            n_ctx: encoder.n_ctx,
            self_k: vec![Vec::new(); n_layer],
            self_v: vec![Vec::new(); n_layer],
            pos: 0,
        })
    }

    /// Clears the self-attention cache (the cross K/V stay valid) so a fresh
    /// decode of the same audio reproduces the first run.
    pub fn reset(&mut self) {
        for k in &mut self.self_k {
            k.clear();
        }
        for v in &mut self.self_v {
            v.clear();
        }
        self.pos = 0;
    }

    /// Number of tokens currently in the self-attention cache.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Advances the decoder by `tokens`, appending their K/V to the cache, and
    /// returns the logits for **every** new token, row-major `[tokens, n_vocab]`.
    ///
    /// The caller reads the last row for greedy / beam expansion; the parity
    /// tests use all rows.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if a token id is out of range or the
    /// decode would exceed `n_text_ctx`.
    pub fn step(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let d = self.cfg.d_model;
        let t = tokens.len();
        if t == 0 {
            return Ok(Vec::new());
        }
        if self.pos + t > self.cfg.n_text_ctx {
            return Err(VokraError::InvalidArgument(format!(
                "whisper decoder: position {} exceeds n_text_ctx {}",
                self.pos + t,
                self.cfg.n_text_ctx
            )));
        }

        // Token + positional embedding.
        let mut h = vec![0.0f32; t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            if tok >= self.cfg.n_vocab {
                return Err(VokraError::InvalidArgument(format!(
                    "whisper decoder: token id {tok} >= n_vocab {}",
                    self.cfg.n_vocab
                )));
            }
            let posidx = self.pos + i;
            let emb = &self.w.token_emb[tok * d..tok * d + d];
            let pe = &self.w.pos_emb[posidx * d..posidx * d + d];
            for c in 0..d {
                h[i * d + c] = emb[c] + pe[c];
            }
        }

        for (li, layer) in self.w.layers.iter().enumerate() {
            // Causal self-attention over the growing cache.
            let normed = layer_norm(&h, t, &layer.self_ln)?;
            let (kh, vh) = project_kv(&normed, t, &layer.self_attn)?;
            self.self_k[li].extend_from_slice(&kh);
            self.self_v[li].extend_from_slice(&vh);
            let t_kv = self.pos + t;
            let attn = attention_from_kv(
                &normed,
                t,
                &self.self_k[li],
                &self.self_v[li],
                t_kv,
                &layer.self_attn.q,
                &layer.self_attn.out,
                self.cfg.n_text_head,
                true,
                self.pos,
            )?;
            add_into(&mut h, &attn)?;

            // Cross-attention over the (fixed) encoder output.
            let normed = layer_norm(&h, t, &layer.cross_ln)?;
            let (ck, cv) = &self.cross_kv[li];
            let attn = attention_from_kv(
                &normed,
                t,
                ck,
                cv,
                self.n_ctx,
                &layer.cross_attn.q,
                &layer.cross_attn.out,
                self.cfg.n_text_head,
                false,
                0,
            )?;
            add_into(&mut h, &attn)?;

            // MLP.
            let normed = layer_norm(&h, t, &layer.mlp_ln)?;
            let ff = mlp(&normed, t, &layer.fc1, &layer.fc2)?;
            add_into(&mut h, &ff)?;
        }

        let h = layer_norm(&h, t, &self.w.ln_post)?;
        self.pos += t;
        project_logits(&h, t, self.cfg, self.w)
    }

    /// Logits for the last token after advancing by `tokens` (greedy / beam).
    pub fn step_last(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let all = self.step(tokens)?;
        let v = self.cfg.n_vocab;
        let t = tokens.len();
        Ok(all[(t - 1) * v..t * v].to_vec())
    }
}

/// `logits[T, n_vocab] = h[T, d] · token_embᵀ` (tied weights, no bias).
///
/// Computed as `token_emb[n_vocab, d] · hᵀ[d, T] → [n_vocab, T]` (so the huge
/// `token_emb` is never transposed), then transposed to `[T, n_vocab]`.
fn project_logits(
    h: &[f32],
    t: usize,
    cfg: &WhisperConfig,
    w: &DecoderWeights,
) -> Result<Vec<f32>> {
    let d = cfg.d_model;
    let v = cfg.n_vocab;
    // hᵀ [d, T].
    let mut h_t = vec![0.0f32; d * t];
    for i in 0..t {
        for c in 0..d {
            h_t[c * t + i] = h[i * d + c];
        }
    }
    // logits_t [v, T] = token_emb [v, d] @ hᵀ [d, T].
    let mut logits_t = vec![0.0f32; v * t];
    gemm_f32(v, t, d, &w.token_emb, &h_t, None, &mut logits_t)?;
    // Transpose to [T, v].
    let mut out = vec![0.0f32; t * v];
    for row in 0..v {
        for col in 0..t {
            out[col * v + row] = logits_t[row * t + col];
        }
    }
    Ok(out)
}
