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

/// Synthetic tiny-decoder builders, shared with the [`super::greedy`] tests so
/// the KV-cache / greedy loops run in CI without a GGUF fixture. Everything is
/// deterministic and small (`d_model = 2`, `n_vocab = 3`); the assertions are
/// internal oracles (error variants, full-vs-cached agreement, determinism),
/// never a reference number.
#[cfg(test)]
pub(crate) mod test_support {
    use crate::whisper::config::WhisperConfig;
    use crate::whisper::encoder::EncoderOutput;
    use crate::whisper::weights::{Attention, DecoderLayer, DecoderWeights, LayerNorm, Linear};

    /// A tiny valid config with `n_layer` decoder blocks (`d_model = 2`,
    /// `n_vocab = 3`, `n_text_ctx = 8`, single head).
    pub(crate) fn tiny_cfg(n_layer: usize) -> WhisperConfig {
        WhisperConfig {
            n_mels: 80,
            d_model: 2,
            n_audio_ctx: 4,
            n_audio_head: 1,
            n_audio_layer: 0,
            n_text_ctx: 8,
            n_text_head: 1,
            n_text_layer: n_layer,
            n_vocab: 3,
            ffn_dim: 2,
            eot: 0,
            decoder_start_ids: vec![1],
        }
    }

    /// Deterministic encoder hidden states `[n_ctx, d_model]`.
    pub(crate) fn tiny_encoder(d_model: usize, n_ctx: usize) -> EncoderOutput {
        let hidden = (0..n_ctx * d_model)
            .map(|i| 0.05 * i as f32 - 0.1)
            .collect();
        EncoderOutput {
            hidden,
            n_ctx,
            d_model,
        }
    }

    /// Decoder weights matching `cfg` (token / pos embeddings, `n_text_layer`
    /// blocks, unit final LayerNorm), all deterministic small values.
    pub(crate) fn tiny_weights(cfg: &WhisperConfig) -> DecoderWeights {
        let d = cfg.d_model;
        let token_emb = (0..cfg.n_vocab * d).map(|i| 0.1 * i as f32 - 0.2).collect();
        let pos_emb = (0..cfg.n_text_ctx * d)
            .map(|i| 0.05 - 0.02 * i as f32)
            .collect();
        let layers = (0..cfg.n_text_layer)
            .map(|_| tiny_layer(d, cfg.ffn_dim))
            .collect();
        DecoderWeights {
            token_emb,
            pos_emb,
            layers,
            ln_post: unit_ln(d),
        }
    }

    /// Unit-scale / zero-shift LayerNorm of width `d`.
    fn unit_ln(d: usize) -> LayerNorm {
        LayerNorm {
            gamma: vec![1.0; d],
            beta: vec![0.0; d],
        }
    }

    /// A `Linear [in, out]` from an explicit row-major `w_t` and optional bias.
    fn lin(
        w_t: Vec<f32>,
        in_features: usize,
        out_features: usize,
        bias: Option<Vec<f32>>,
    ) -> Linear {
        Linear {
            w_t,
            in_features,
            out_features,
            bias,
        }
    }

    /// A deterministic `rows * cols` weight buffer with small distinct values.
    fn rect(rows: usize, cols: usize, base: f32) -> Vec<f32> {
        (0..rows * cols).map(|i| base + 0.03 * i as f32).collect()
    }

    /// A `[d, d]` attention block with deterministic projections (`k` has no
    /// bias, matching Whisper).
    fn tiny_attn(d: usize) -> Attention {
        Attention {
            q: lin(rect(d, d, 0.10), d, d, Some(vec![0.01; d])),
            k: lin(rect(d, d, -0.07), d, d, None),
            v: lin(rect(d, d, 0.05), d, d, Some(vec![0.02; d])),
            out: lin(rect(d, d, -0.04), d, d, Some(vec![0.0; d])),
        }
    }

    /// One decoder block (self-attn → cross-attn → MLP) with deterministic
    /// weights; `ff` is the MLP inner width.
    fn tiny_layer(d: usize, ff: usize) -> DecoderLayer {
        DecoderLayer {
            self_ln: unit_ln(d),
            self_attn: tiny_attn(d),
            cross_ln: unit_ln(d),
            cross_attn: tiny_attn(d),
            mlp_ln: unit_ln(d),
            fc1: lin(rect(d, ff, 0.06), d, ff, Some(vec![0.0; ff])),
            fc2: lin(rect(ff, d, -0.05), ff, d, Some(vec![0.0; d])),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{tiny_cfg, tiny_encoder, tiny_weights};
    use super::*;

    #[test]
    fn new_rejects_encoder_dim_mismatch() {
        let cfg = tiny_cfg(0);
        let w = tiny_weights(&cfg);
        // Encoder hidden width differs from the config d_model. (DecoderState is
        // not Debug, so match instead of unwrap_err.)
        let enc = tiny_encoder(cfg.d_model + 1, 4);
        assert!(matches!(
            DecoderState::new(&cfg, &w, &enc),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn step_rejects_out_of_range_token() {
        let cfg = tiny_cfg(0);
        let w = tiny_weights(&cfg);
        let enc = tiny_encoder(cfg.d_model, 4);
        let mut st = DecoderState::new(&cfg, &w, &enc).unwrap();
        // 99 >= n_vocab (3): guarded before the embedding slice would panic.
        let err = st.step(&[99]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn step_rejects_exceeding_n_text_ctx() {
        let cfg = tiny_cfg(0);
        let w = tiny_weights(&cfg);
        let enc = tiny_encoder(cfg.d_model, 4);
        let mut st = DecoderState::new(&cfg, &w, &enc).unwrap();
        // n_text_ctx = 8; an n_text_ctx+1 step overflows (ids stay in vocab).
        let toks = vec![1u32; cfg.n_text_ctx + 1];
        let err = st.step(&toks).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn empty_step_returns_empty_and_does_not_advance() {
        let cfg = tiny_cfg(0);
        let w = tiny_weights(&cfg);
        let enc = tiny_encoder(cfg.d_model, 4);
        let mut st = DecoderState::new(&cfg, &w, &enc).unwrap();
        assert!(st.step(&[]).unwrap().is_empty());
        assert_eq!(st.position(), 0);
    }

    /// Full-sequence `step` must reproduce the token-by-token cached path for
    /// the last position — the KV-cache invariant the parity test owns at real
    /// scale (verified here at synthetic scale, both with and without a layer).
    fn assert_full_matches_cached(n_layer: usize) {
        let cfg = tiny_cfg(n_layer);
        let w = tiny_weights(&cfg);
        let enc = tiny_encoder(cfg.d_model, 4);
        let v = cfg.n_vocab;
        let (a, b) = (1u32, 2u32);

        let mut st = DecoderState::new(&cfg, &w, &enc).unwrap();
        let full = st.step(&[a, b]).unwrap();
        assert_eq!(full.len(), 2 * v);
        let full_last = &full[v..2 * v];

        // Token-by-token after reset must land on the same last-position logits.
        st.reset();
        assert_eq!(st.position(), 0);
        let _ = st.step_last(&[a]).unwrap();
        let cached_last = st.step_last(&[b]).unwrap();
        assert_eq!(cached_last.len(), v);
        for (i, (&f, &c)) in full_last.iter().zip(&cached_last).enumerate() {
            assert!(
                (f - c).abs() < 1e-4,
                "layer {n_layer} idx {i}: full {f} vs cached {c}"
            );
        }
    }

    #[test]
    fn full_forward_matches_cached_stepping() {
        assert_full_matches_cached(0);
        assert_full_matches_cached(1);
    }

    #[test]
    fn reset_and_replay_is_bit_identical() {
        let cfg = tiny_cfg(1);
        let w = tiny_weights(&cfg);
        let enc = tiny_encoder(cfg.d_model, 4);
        let mut st = DecoderState::new(&cfg, &w, &enc).unwrap();

        let run1 = st.step(&[1, 2, 1]).unwrap();
        st.reset();
        let run2 = st.step(&[1, 2, 1]).unwrap();
        // Same code path, same inputs, same accumulation order → bit-identical.
        assert_eq!(run1, run2);
    }

    #[test]
    fn step_last_is_final_row_of_step() {
        let cfg = tiny_cfg(1);
        let w = tiny_weights(&cfg);
        let enc = tiny_encoder(cfg.d_model, 4);
        let v = cfg.n_vocab;

        let mut st = DecoderState::new(&cfg, &w, &enc).unwrap();
        let all = st.step(&[1, 2]).unwrap();
        let last_slice = all[v..2 * v].to_vec();

        st.reset();
        let last = st.step_last(&[1, 2]).unwrap();
        assert_eq!(last, last_slice);
    }
}
