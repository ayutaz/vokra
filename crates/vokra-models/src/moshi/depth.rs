//! Moshi depformer — per-frame codebook-axis autoregression with
//! **per-step weights** (M4-06-T11).
//!
//! # Architecture (ADR M4-06 §D2 — transcribed, never invented)
//!
//! `kyutai-labs/moshi` `lm.py` `depformer_step` / `forward_depformer` +
//! `transformer.py` `apply_weights_per_step`:
//!
//! - `dep_q` sequential steps per 12.5 Hz frame; step `cb` consumes
//!   `depformer_in[cb](temporal_hidden) + emb(prev_token)` where the
//!   previous token is the **text token** at `cb == 0`
//!   (`depformer_text_emb`) and the previously sampled audio code
//!   otherwise (`depformer_emb[cb-1]`);
//! - `n_layer` pre-norm blocks whose attention in/out projections and
//!   gating FFNs carry **one weight set per step** (`weights_per_step =
//!   dep_q` — the T02 manifest shows `in_proj_weight` at `[dep_q·3·d, d]`
//!   etc.); `norm1` / `norm2` are shared across steps;
//! - **no positional embedding** (`depformer_pos_emb="none"`), plain
//!   causal attention over the ≤ `dep_q` frame-local positions, KV
//!   **reset every frame** (`set_streaming_detached` upstream);
//! - `linears[cb]` (`[card, d]`) reads the final hidden (depformer_norms
//!   are `Identity` upstream — no extra norm before the head).
//!
//! # KV: deliberately not paged (ADR M4-06 §D1-(e))
//!
//! At most `dep_q` positions, reset per frame — a fixed pre-allocated
//! scratch carries it with zero allocation in the frame loop (FR-EX-05;
//! the CSM §D4 arithmetic).

use vokra_core::gguf::GgufFile;
use vokra_core::rng::SplitMix64;
use vokra_core::{BackendKind, Result, VokraError};

use super::backbone::{MOSHI_HOT_OPS, tensor_f32, transpose};
use super::config::MoshiConfig;
use crate::compute::Compute;
use crate::csm::backbone::xavier_uniform;
use crate::voxtral::text_decoder::{hadamard_inplace, rms_norm, silu_inplace};

/// One depformer layer: shared norms + per-step attention / gating
/// weights (`Vec` index = codebook step; every `w_t` is the Compute-seam
/// `[in, out]` transposed layout).
#[derive(Debug, Clone)]
pub struct MoshiDepthLayer {
    /// Shared attention RMSNorm γ `[d]` (`norm1.alpha`).
    pub norm1_gamma: Vec<f32>,
    /// Shared FFN RMSNorm γ `[d]` (`norm2.alpha`).
    pub norm2_gamma: Vec<f32>,
    /// Per-step Q projections `[d, d]` transposed.
    pub q_w_t: Vec<Vec<f32>>,
    /// Per-step K projections.
    pub k_w_t: Vec<Vec<f32>>,
    /// Per-step V projections.
    pub v_w_t: Vec<Vec<f32>>,
    /// Per-step output projections.
    pub o_w_t: Vec<Vec<f32>>,
    /// Per-step gating gate halves `[d, hidden]` transposed.
    pub gate_w_t: Vec<Vec<f32>>,
    /// Per-step gating up halves.
    pub up_w_t: Vec<Vec<f32>>,
    /// Per-step gating down projections `[hidden, d]` transposed.
    pub down_w_t: Vec<Vec<f32>>,
}

/// All depformer weights (ADR M4-06 §D2 manifest table).
#[derive(Debug, Clone)]
pub struct MoshiDepthWeights {
    /// Per-step temporal→depth projections `[d_temporal, d_depth]`
    /// transposed (`depformer_in.{cb}.weight`, `depformer_multi_linear`).
    pub depformer_in_w_t: Vec<Vec<f32>>,
    /// Text conditioning embedding `[(text_card + 1) * d_depth]`
    /// (`depformer_text_emb.weight`).
    pub text_emb: Vec<f32>,
    /// Audio conditioning embeddings, step `cb >= 1` reads table `cb - 1`
    /// — `[(dep_q - 1) * (audio_card + 1) * d_depth]`
    /// (`depformer_emb.{cb-1}.weight`).
    pub audio_emb: Vec<f32>,
    /// Transformer layers (per-step weights inside).
    pub layers: Vec<MoshiDepthLayer>,
    /// Per-step output heads `[card, d_depth]` row-major GEMV layout
    /// (`linears.{cb}.weight` verbatim).
    pub heads: Vec<Vec<f32>>,
    /// `true` when built by [`Self::synthesized`].
    pub is_synthesized: bool,
}

impl MoshiDepthWeights {
    /// Synthesized (seed-deterministic) store — Xavier projections, γ = 1.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on an ill-formed config.
    pub fn synthesized(config: &MoshiConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let d_t = config.temporal.d_model;
        let d = config.depth.d_model;
        let h = config.depth.ffn_hidden;
        let dep_q = config.dep_q;
        let mut rng = SplitMix64::new(seed);
        let per_step =
            |rng: &mut SplitMix64, n: usize, fi: usize, fo: usize, steps: usize| -> Vec<Vec<f32>> {
                (0..steps).map(|_| xavier_uniform(rng, n, fi, fo)).collect()
            };
        let depformer_in_w_t = per_step(&mut rng, d_t * d, d_t, d, dep_q);
        let text_emb = xavier_uniform(&mut rng, (config.text_card + 1) * d, d, d);
        let audio_emb = xavier_uniform(
            &mut rng,
            dep_q.saturating_sub(1) * (config.audio_card + 1) * d,
            d,
            d,
        );
        let mut layers = Vec::with_capacity(config.depth.n_layer);
        for _ in 0..config.depth.n_layer {
            layers.push(MoshiDepthLayer {
                norm1_gamma: vec![1.0f32; d],
                norm2_gamma: vec![1.0f32; d],
                q_w_t: per_step(&mut rng, d * d, d, d, dep_q),
                k_w_t: per_step(&mut rng, d * d, d, d, dep_q),
                v_w_t: per_step(&mut rng, d * d, d, d, dep_q),
                o_w_t: per_step(&mut rng, d * d, d, d, dep_q),
                gate_w_t: per_step(&mut rng, d * h, d, h, dep_q),
                up_w_t: per_step(&mut rng, d * h, d, h, dep_q),
                down_w_t: per_step(&mut rng, h * d, h, d, dep_q),
            });
        }
        let heads = per_step(&mut rng, config.audio_card * d, d, config.audio_card, dep_q);
        Ok(Self {
            depformer_in_w_t,
            text_emb,
            audio_emb,
            layers,
            heads,
            is_synthesized: true,
        })
    }

    /// Binds real weights from a Moshi GGUF. The packed per-step tensors
    /// (`in_proj_weight` `[dep_q·3·d, d]`, `out_proj.weight`
    /// `[dep_q·d, d]`) are split per the upstream `_load_hook`
    /// `view(mult, -1, in)` convention (ADR M4-06 §D2); the per-step
    /// gating tensors are separate upstream (`gating.{s}.linear_in`).
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming any missing / mis-shaped tensor.
    pub fn from_gguf(file: &GgufFile, config: &MoshiConfig) -> Result<Self> {
        config.validate_for_forward()?;
        let d_t = config.temporal.d_model;
        let d = config.depth.d_model;
        let h = config.depth.ffn_hidden;
        let dep_q = config.dep_q;

        let mut depformer_in_w_t = Vec::with_capacity(dep_q);
        for cb in 0..dep_q {
            let t = tensor_f32(file, &format!("depformer_in.{cb}.weight"), d * d_t)?;
            depformer_in_w_t.push(transpose(&t, d, d_t));
        }
        let text_emb = tensor_f32(
            file,
            "depformer_text_emb.weight",
            (config.text_card + 1) * d,
        )?;
        let mut audio_emb =
            Vec::with_capacity(dep_q.saturating_sub(1) * (config.audio_card + 1) * d);
        for cb in 0..dep_q.saturating_sub(1) {
            let t = tensor_f32(
                file,
                &format!("depformer_emb.{cb}.weight"),
                (config.audio_card + 1) * d,
            )?;
            audio_emb.extend_from_slice(&t);
        }
        let mut layers = Vec::with_capacity(config.depth.n_layer);
        for i in 0..config.depth.n_layer {
            let p = format!("depformer.layers.{i}");
            // Packed per-step QKV: [dep_q · 3d, d] — step-major
            // (`view(mult, -1, in)`), then Q/K/V thirds inside each step.
            let in_proj = tensor_f32(
                file,
                &format!("{p}.self_attn.in_proj_weight"),
                dep_q * 3 * d * d,
            )?;
            let out_proj = tensor_f32(
                file,
                &format!("{p}.self_attn.out_proj.weight"),
                dep_q * d * d,
            )?;
            let mut q_w_t = Vec::with_capacity(dep_q);
            let mut k_w_t = Vec::with_capacity(dep_q);
            let mut v_w_t = Vec::with_capacity(dep_q);
            let mut o_w_t = Vec::with_capacity(dep_q);
            for s in 0..dep_q {
                let base = s * 3 * d * d;
                q_w_t.push(transpose(&in_proj[base..base + d * d], d, d));
                k_w_t.push(transpose(&in_proj[base + d * d..base + 2 * d * d], d, d));
                v_w_t.push(transpose(
                    &in_proj[base + 2 * d * d..base + 3 * d * d],
                    d,
                    d,
                ));
                o_w_t.push(transpose(&out_proj[s * d * d..(s + 1) * d * d], d, d));
            }
            let mut gate_w_t = Vec::with_capacity(dep_q);
            let mut up_w_t = Vec::with_capacity(dep_q);
            let mut down_w_t = Vec::with_capacity(dep_q);
            for s in 0..dep_q {
                let lin_in =
                    tensor_f32(file, &format!("{p}.gating.{s}.linear_in.weight"), 2 * h * d)?;
                gate_w_t.push(transpose(&lin_in[0..h * d], h, d));
                up_w_t.push(transpose(&lin_in[h * d..2 * h * d], h, d));
                let lin_out =
                    tensor_f32(file, &format!("{p}.gating.{s}.linear_out.weight"), d * h)?;
                down_w_t.push(transpose(&lin_out, d, h));
            }
            layers.push(MoshiDepthLayer {
                norm1_gamma: tensor_f32(file, &format!("{p}.norm1.alpha"), d)?,
                norm2_gamma: tensor_f32(file, &format!("{p}.norm2.alpha"), d)?,
                q_w_t,
                k_w_t,
                v_w_t,
                o_w_t,
                gate_w_t,
                up_w_t,
                down_w_t,
            });
        }
        let mut heads = Vec::with_capacity(dep_q);
        for cb in 0..dep_q {
            heads.push(tensor_f32(
                file,
                &format!("linears.{cb}.weight"),
                config.audio_card * d,
            )?);
        }
        Ok(Self {
            depformer_in_w_t,
            text_emb,
            audio_emb,
            layers,
            heads,
            is_synthesized: false,
        })
    }
}

/// Per-frame depformer state: fixed-capacity per-layer K/V (`dep_q`
/// positions), reset each frame with zero allocation.
pub struct MoshiDepthState {
    seq_len: usize,
    /// Per-layer K cache `[n_layer][dep_q * d]` flattened.
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    // Step scratch (t = 1).
    h: Vec<f32>,
    norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    scores: Vec<f32>,
    probs: Vec<f32>,
    attn_out: Vec<f32>,
    attn_o: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
    proj_in: Vec<f32>,
    logits: Vec<f32>,
}

impl std::fmt::Debug for MoshiDepthState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoshiDepthState")
            .field("seq_len", &self.seq_len)
            .finish()
    }
}

impl MoshiDepthState {
    /// Pre-allocates the per-frame scratch for `config`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on an ill-formed config.
    pub fn new(config: &MoshiConfig) -> Result<Self> {
        config.validate_for_forward()?;
        let d = config.depth.d_model;
        let h = config.depth.ffn_hidden;
        let max_pos = config.dep_q;
        let n_layer = config.depth.n_layer;
        Ok(Self {
            seq_len: 0,
            k_cache: vec![0.0; n_layer * max_pos * d],
            v_cache: vec![0.0; n_layer * max_pos * d],
            h: vec![0.0; d],
            norm: vec![0.0; d],
            q: vec![0.0; d],
            k: vec![0.0; d],
            v: vec![0.0; d],
            scores: vec![0.0; max_pos],
            probs: vec![0.0; max_pos],
            attn_out: vec![0.0; d],
            attn_o: vec![0.0; d],
            ffn_gate: vec![0.0; h],
            ffn_up: vec![0.0; h],
            ffn_down: vec![0.0; d],
            proj_in: vec![0.0; d],
            logits: vec![0.0; config.audio_card],
        })
    }

    /// Rewinds to an empty frame (upstream per-frame streaming reset).
    /// No allocation — the scratch is retained.
    pub fn begin_frame(&mut self) {
        self.seq_len = 0;
    }

    /// Steps decoded in the current frame.
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
}

/// The depformer (config + weights + backend).
pub struct MoshiDepthTransformer {
    config: MoshiConfig,
    weights: MoshiDepthWeights,
    backend: BackendKind,
}

impl std::fmt::Debug for MoshiDepthTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoshiDepthTransformer")
            .field("depth", &self.config.depth)
            .field("dep_q", &self.config.dep_q)
            .field("weights.is_synthesized", &self.weights.is_synthesized)
            .field("backend", &self.backend)
            .finish()
    }
}

impl MoshiDepthTransformer {
    /// Builds the depformer from an explicit weight store.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on config / shape mismatch.
    pub fn new(config: MoshiConfig, weights: MoshiDepthWeights) -> Result<Self> {
        config.validate_for_forward()?;
        validate_shapes(&config, &weights)?;
        Ok(Self {
            config,
            weights,
            backend: BackendKind::Cpu,
        })
    }

    /// Synthesized-fixture constructor.
    ///
    /// # Errors
    ///
    /// Propagates [`MoshiDepthWeights::synthesized`].
    pub fn synthesized(config: MoshiConfig, seed: u64) -> Result<Self> {
        let weights = MoshiDepthWeights::synthesized(&config, seed)?;
        Self::new(config, weights)
    }

    /// Selects the Compute-seam backend.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// The weight store.
    #[must_use]
    pub fn weights(&self) -> &MoshiDepthWeights {
        &self.weights
    }

    /// The resolved config (cross-stack coherence checks —
    /// `MoshiModel::from_parts`).
    #[must_use]
    pub fn config(&self) -> &MoshiConfig {
        &self.config
    }

    fn compute(&self) -> Result<Compute> {
        Compute::for_backend(self.backend, MOSHI_HOT_OPS)
    }

    /// Decodes the `dep_q` own-audio codebooks for one frame, conditioned
    /// on the temporal hidden state and the just-sampled text token
    /// (lm.py `depformer_step`). `codes_out[cb]` receives step `cb`'s
    /// sampled id; `sample` gets the `[card]` logits row (M1 sampler
    /// closure shape).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on shape mismatches, an
    /// out-of-range text token, or a sampler returning `>= card`
    /// (FR-EX-08 — a wrong id would corrupt the RVQ stream).
    pub fn decode_frame(
        &self,
        temporal_hidden: &[f32],
        text_token: u32,
        state: &mut MoshiDepthState,
        codes_out: &mut [u32],
        mut sample: impl FnMut(&mut [f32]) -> u32,
    ) -> Result<()> {
        let cfg = &self.config;
        let d_t = cfg.temporal.d_model;
        let d = cfg.depth.d_model;
        if temporal_hidden.len() != d_t {
            return Err(VokraError::InvalidArgument(format!(
                "moshi depformer: temporal_hidden len {} != d_temporal {d_t}",
                temporal_hidden.len()
            )));
        }
        if codes_out.len() != cfg.dep_q {
            return Err(VokraError::InvalidArgument(format!(
                "moshi depformer: codes_out len {} != dep_q {}",
                codes_out.len(),
                cfg.dep_q
            )));
        }
        if text_token as usize > cfg.text_card {
            return Err(VokraError::InvalidArgument(format!(
                "moshi depformer: text token {text_token} >= text rows {}",
                cfg.text_card + 1
            )));
        }
        let compute = self.compute()?;
        state.begin_frame();

        let card = cfg.audio_card as u32;
        for cb in 0..cfg.dep_q {
            // input = depformer_in[cb](temporal_hidden) + emb(prev).
            compute.gemm_f32(
                1,
                d,
                d_t,
                temporal_hidden,
                &self.weights.depformer_in_w_t[cb],
                None,
                &mut state.proj_in,
            )?;
            if cb == 0 {
                let row = text_token as usize * d;
                for (dst, src) in state
                    .proj_in
                    .iter_mut()
                    .zip(&self.weights.text_emb[row..row + d])
                {
                    *dst += *src;
                }
            } else {
                let prev = codes_out[cb - 1] as usize;
                // prev < card is guaranteed below; the audio conditioning
                // tables carry card + 1 rows like the temporal ones.
                let base = ((cb - 1) * (cfg.audio_card + 1) + prev) * d;
                for (dst, src) in state
                    .proj_in
                    .iter_mut()
                    .zip(&self.weights.audio_emb[base..base + d])
                {
                    *dst += *src;
                }
            }
            state.h.copy_from_slice(&state.proj_in);
            self.step(&compute, cb, state)?;
            // Head cb reads the final hidden directly (depformer_norms =
            // Identity upstream).
            compute.gemv_f32(
                cfg.audio_card,
                d,
                &self.weights.heads[cb],
                &state.h,
                None,
                &mut state.logits,
            )?;
            let code = sample(&mut state.logits);
            if code >= card {
                return Err(VokraError::InvalidArgument(format!(
                    "moshi depformer: sampled code {code} >= card {card} (sampler \
                     misconfigured — FR-EX-08)"
                )));
            }
            codes_out[cb] = code;
        }
        Ok(())
    }

    /// One frame-local transformer position at step index `step` (selects
    /// the per-step weight set — `apply_weights_per_step`). Appends K/V at
    /// position `state.seq_len` and advances it.
    fn step(&self, compute: &Compute, step: usize, state: &mut MoshiDepthState) -> Result<()> {
        let cfg = &self.config.depth;
        let d = cfg.d_model;
        let n_head = cfg.n_head;
        let head_dim = cfg.head_dim();
        let h_ffn = cfg.ffn_hidden;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let eps = self.config.rms_norm_eps;
        let max_pos = self.config.dep_q;
        let pos = state.seq_len;
        if pos >= max_pos {
            return Err(VokraError::InvalidArgument(format!(
                "moshi depformer: frame-local position {pos} >= dep_q {max_pos} \
                 (begin_frame not called?)"
            )));
        }

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            // ---------- Pre-norm MHA (per-step weights, no RoPE) ----------
            rms_norm(&state.h, &layer.norm1_gamma, eps, 1, &mut state.norm)?;
            compute.gemm_f32(1, d, d, &state.norm, &layer.q_w_t[step], None, &mut state.q)?;
            compute.gemm_f32(1, d, d, &state.norm, &layer.k_w_t[step], None, &mut state.k)?;
            compute.gemm_f32(1, d, d, &state.norm, &layer.v_w_t[step], None, &mut state.v)?;

            let base = layer_idx * max_pos * d;
            state.k_cache[base + pos * d..base + (pos + 1) * d].copy_from_slice(&state.k);
            state.v_cache[base + pos * d..base + (pos + 1) * d].copy_from_slice(&state.v);

            let t_kv = pos + 1;
            for h_i in 0..n_head {
                let q_row = &state.q[h_i * head_dim..(h_i + 1) * head_dim];
                for j in 0..t_kv {
                    let k_row = &state.k_cache
                        [base + j * d + h_i * head_dim..base + j * d + (h_i + 1) * head_dim];
                    let mut s = 0.0f32;
                    for c in 0..head_dim {
                        s += q_row[c] * k_row[c];
                    }
                    state.scores[j] = s * scale;
                }
                compute.softmax_f32(&state.scores[..t_kv], &mut state.probs[..t_kv], 1, t_kv)?;
                let out_dst = &mut state.attn_out[h_i * head_dim..(h_i + 1) * head_dim];
                for (c, out) in out_dst.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for j in 0..t_kv {
                        sum += state.probs[j] * state.v_cache[base + j * d + h_i * head_dim + c];
                    }
                    *out = sum;
                }
            }
            compute.gemm_f32(
                1,
                d,
                d,
                &state.attn_out,
                &layer.o_w_t[step],
                None,
                &mut state.attn_o,
            )?;
            for i in 0..d {
                state.h[i] += state.attn_o[i];
            }

            // ---------- Pre-norm SiLU-gating FFN (per-step weights) ----------
            rms_norm(&state.h, &layer.norm2_gamma, eps, 1, &mut state.norm)?;
            compute.gemm_f32(
                1,
                h_ffn,
                d,
                &state.norm,
                &layer.gate_w_t[step],
                None,
                &mut state.ffn_gate,
            )?;
            compute.gemm_f32(
                1,
                h_ffn,
                d,
                &state.norm,
                &layer.up_w_t[step],
                None,
                &mut state.ffn_up,
            )?;
            silu_inplace(&mut state.ffn_gate);
            hadamard_inplace(&mut state.ffn_gate, &state.ffn_up)?;
            compute.gemm_f32(
                1,
                d,
                h_ffn,
                &state.ffn_gate,
                &layer.down_w_t[step],
                None,
                &mut state.ffn_down,
            )?;
            for i in 0..d {
                state.h[i] += state.ffn_down[i];
            }
        }
        state.seq_len += 1;
        Ok(())
    }
}

fn validate_shapes(config: &MoshiConfig, weights: &MoshiDepthWeights) -> Result<()> {
    let d_t = config.temporal.d_model;
    let d = config.depth.d_model;
    let h = config.depth.ffn_hidden;
    let dep_q = config.dep_q;
    if weights.depformer_in_w_t.len() != dep_q
        || weights.heads.len() != dep_q
        || weights.layers.len() != config.depth.n_layer
    {
        return Err(VokraError::InvalidArgument(format!(
            "moshi depformer: per-step counts (in {}, heads {}, layers {}) don't \
             match dep_q {} / n_layer {}",
            weights.depformer_in_w_t.len(),
            weights.heads.len(),
            weights.layers.len(),
            dep_q,
            config.depth.n_layer
        )));
    }
    for (cb, w) in weights.depformer_in_w_t.iter().enumerate() {
        if w.len() != d_t * d {
            return Err(VokraError::InvalidArgument(format!(
                "moshi depformer: depformer_in[{cb}] len {} != {}",
                w.len(),
                d_t * d
            )));
        }
    }
    if weights.text_emb.len() != (config.text_card + 1) * d {
        return Err(VokraError::InvalidArgument(format!(
            "moshi depformer: text_emb len {} != {}",
            weights.text_emb.len(),
            (config.text_card + 1) * d
        )));
    }
    if weights.audio_emb.len() != dep_q.saturating_sub(1) * (config.audio_card + 1) * d {
        return Err(VokraError::InvalidArgument(format!(
            "moshi depformer: audio_emb len {} != {}",
            weights.audio_emb.len(),
            dep_q.saturating_sub(1) * (config.audio_card + 1) * d
        )));
    }
    for (i, l) in weights.layers.iter().enumerate() {
        if l.norm1_gamma.len() != d || l.norm2_gamma.len() != d {
            return Err(VokraError::InvalidArgument(format!(
                "moshi depformer: layer[{i}] norm γ lens ({}, {}) != d {d}",
                l.norm1_gamma.len(),
                l.norm2_gamma.len()
            )));
        }
        for (name, per_step, want) in [
            ("q_w_t", &l.q_w_t, d * d),
            ("k_w_t", &l.k_w_t, d * d),
            ("v_w_t", &l.v_w_t, d * d),
            ("o_w_t", &l.o_w_t, d * d),
            ("gate_w_t", &l.gate_w_t, d * h),
            ("up_w_t", &l.up_w_t, d * h),
            ("down_w_t", &l.down_w_t, h * d),
        ] {
            if per_step.len() != dep_q {
                return Err(VokraError::InvalidArgument(format!(
                    "moshi depformer: layer[{i}].{name} has {} step sets, want {dep_q}",
                    per_step.len()
                )));
            }
            for (s, w) in per_step.iter().enumerate() {
                if w.len() != want {
                    return Err(VokraError::InvalidArgument(format!(
                        "moshi depformer: layer[{i}].{name}[{s}] len {} != {want}",
                        w.len()
                    )));
                }
            }
        }
    }
    for (cb, head) in weights.heads.iter().enumerate() {
        if head.len() != config.audio_card * d {
            return Err(VokraError::InvalidArgument(format!(
                "moshi depformer: heads[{cb}] len {} != {}",
                head.len(),
                config.audio_card * d
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::decode::argmax;

    fn depth() -> MoshiDepthTransformer {
        MoshiDepthTransformer::synthesized(MoshiConfig::tiny_for_tests(), 31).expect("depformer")
    }

    fn hidden(cfg: &MoshiConfig, seed: u32) -> Vec<f32> {
        (0..cfg.temporal.d_model)
            .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 0.4)
            .collect()
    }

    #[test]
    fn decode_frame_yields_dep_q_codes_in_range() {
        let m = depth();
        let cfg = MoshiConfig::tiny_for_tests();
        let mut state = MoshiDepthState::new(&cfg).unwrap();
        let mut codes = vec![0u32; cfg.dep_q];
        m.decode_frame(&hidden(&cfg, 1), 2, &mut state, &mut codes, |l| argmax(l))
            .unwrap();
        assert!(codes.iter().all(|&c| (c as usize) < cfg.audio_card));
        assert_eq!(state.seq_len(), cfg.dep_q);
    }

    #[test]
    fn decode_frame_is_deterministic_and_text_conditioned() {
        let m = depth();
        let cfg = MoshiConfig::tiny_for_tests();
        let run = |text_tok: u32| {
            let mut state = MoshiDepthState::new(&cfg).unwrap();
            let mut codes = vec![0u32; cfg.dep_q];
            m.decode_frame(&hidden(&cfg, 3), text_tok, &mut state, &mut codes, |l| {
                argmax(l)
            })
            .unwrap();
            codes
        };
        assert_eq!(run(1), run(1), "greedy decode reproducible");
        // The text token conditions step 0 (depformer_text_emb) — with
        // synthesized weights two different tokens must not always
        // coincide across every codebook (probabilistic but with tiny
        // vocab + random weights the argmax flips; assert on a set).
        let a = run(1);
        let outcomes: std::collections::BTreeSet<Vec<u32>> =
            (0..cfg.text_card as u32).map(run).collect();
        assert!(
            outcomes.len() > 1,
            "text conditioning must reach the codes (all {} tokens produced {a:?})",
            cfg.text_card
        );
    }

    #[test]
    fn per_step_weights_are_actually_selected() {
        // Zero out step 1's head; step 1's logits become the zero vector
        // → argmax = 0 regardless of the hidden state, while step 0 is
        // unaffected. This pins the per-step selection (a shared-weights
        // regression would leak step 0's head into step 1).
        let cfg = MoshiConfig::tiny_for_tests();
        let mut w = MoshiDepthWeights::synthesized(&cfg, 5).unwrap();
        for v in w.heads[1].iter_mut() {
            *v = 0.0;
        }
        // Make step 0's head strongly prefer a non-zero id.
        for v in w.heads[0].iter_mut() {
            *v = 0.0;
        }
        let d = cfg.depth.d_model;
        for c in 0..d {
            w.heads[0][3 * d + c] = 1.0; // row 3 wins for any positive-ish h
        }
        let m = MoshiDepthTransformer::new(cfg.clone(), w).unwrap();
        let mut state = MoshiDepthState::new(&cfg).unwrap();
        let mut codes = vec![0u32; cfg.dep_q];
        // A hidden state whose step-0 final h has positive coordinates is
        // not guaranteed; instead assert the *difference*: step 1 must be
        // 0 (zero head), and the overall decode must not error.
        m.decode_frame(&hidden(&cfg, 2), 1, &mut state, &mut codes, |l| argmax(l))
            .unwrap();
        assert_eq!(codes[1], 0, "zeroed step-1 head → argmax 0");
    }

    #[test]
    fn sampler_out_of_range_is_a_loud_error() {
        let m = depth();
        let cfg = MoshiConfig::tiny_for_tests();
        let mut state = MoshiDepthState::new(&cfg).unwrap();
        let mut codes = vec![0u32; cfg.dep_q];
        let err = m
            .decode_frame(&hidden(&cfg, 1), 0, &mut state, &mut codes, |_| {
                cfg.audio_card as u32
            })
            .unwrap_err();
        assert!(err.to_string().contains("sampler"), "actionable: {err}");
    }

    #[test]
    fn shape_and_argument_errors_are_loud() {
        let m = depth();
        let cfg = MoshiConfig::tiny_for_tests();
        let mut state = MoshiDepthState::new(&cfg).unwrap();
        let mut codes = vec![0u32; cfg.dep_q];
        // Wrong hidden width.
        assert!(
            m.decode_frame(
                &hidden(&cfg, 0)[1..],
                0,
                &mut state,
                &mut codes,
                |l| argmax(l)
            )
            .is_err()
        );
        // Wrong codes arity.
        let mut short = vec![0u32; cfg.dep_q - 1];
        assert!(
            m.decode_frame(&hidden(&cfg, 0), 0, &mut state, &mut short, |l| argmax(l))
                .is_err()
        );
        // Out-of-range text token.
        assert!(
            m.decode_frame(
                &hidden(&cfg, 0),
                cfg.text_initial_token() + 1,
                &mut state,
                &mut codes,
                |l| argmax(l)
            )
            .is_err()
        );
    }

    #[test]
    fn begin_frame_resets_with_zero_alloc_semantics() {
        let m = depth();
        let cfg = MoshiConfig::tiny_for_tests();
        let mut state = MoshiDepthState::new(&cfg).unwrap();
        let mut a = vec![0u32; cfg.dep_q];
        m.decode_frame(&hidden(&cfg, 9), 1, &mut state, &mut a, |l| argmax(l))
            .unwrap();
        let mut b = vec![0u32; cfg.dep_q];
        m.decode_frame(&hidden(&cfg, 9), 1, &mut state, &mut b, |l| argmax(l))
            .unwrap();
        assert_eq!(
            a, b,
            "decode_frame begins its own frame — stateless across calls"
        );
    }

    #[test]
    fn from_gguf_binds_packed_per_step_tensors_round_trip() {
        use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};
        let cfg = MoshiConfig::tiny_for_tests();
        let src = MoshiDepthTransformer::synthesized(cfg.clone(), 77).unwrap();
        let w = src.weights();
        let d_t = cfg.temporal.d_model;
        let d = cfg.depth.d_model;
        let h = cfg.depth.ffn_hidden;
        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "moshi");
        for cb in 0..cfg.dep_q {
            b.add_tensor(
                &format!("depformer_in.{cb}.weight"),
                GgmlType::F32,
                vec![d as u64, d_t as u64],
                f32_bytes(&transpose(&w.depformer_in_w_t[cb], d_t, d)),
            )
            .unwrap();
            b.add_tensor(
                &format!("linears.{cb}.weight"),
                GgmlType::F32,
                vec![cfg.audio_card as u64, d as u64],
                f32_bytes(&w.heads[cb]),
            )
            .unwrap();
        }
        b.add_tensor(
            "depformer_text_emb.weight",
            GgmlType::F32,
            vec![(cfg.text_card + 1) as u64, d as u64],
            f32_bytes(&w.text_emb),
        )
        .unwrap();
        let rows = cfg.audio_card + 1;
        for cb in 0..cfg.dep_q - 1 {
            b.add_tensor(
                &format!("depformer_emb.{cb}.weight"),
                GgmlType::F32,
                vec![rows as u64, d as u64],
                f32_bytes(&w.audio_emb[cb * rows * d..(cb + 1) * rows * d]),
            )
            .unwrap();
        }
        for (i, layer) in w.layers.iter().enumerate() {
            let p = format!("depformer.layers.{i}");
            // Pack per-step QKV back into [dep_q·3d, d] step-major.
            let mut in_proj = Vec::with_capacity(cfg.dep_q * 3 * d * d);
            let mut out_proj = Vec::with_capacity(cfg.dep_q * d * d);
            for s in 0..cfg.dep_q {
                in_proj.extend(transpose(&layer.q_w_t[s], d, d));
                in_proj.extend(transpose(&layer.k_w_t[s], d, d));
                in_proj.extend(transpose(&layer.v_w_t[s], d, d));
                out_proj.extend(transpose(&layer.o_w_t[s], d, d));
            }
            b.add_tensor(
                &format!("{p}.self_attn.in_proj_weight"),
                GgmlType::F32,
                vec![(cfg.dep_q * 3 * d) as u64, d as u64],
                f32_bytes(&in_proj),
            )
            .unwrap();
            b.add_tensor(
                &format!("{p}.self_attn.out_proj.weight"),
                GgmlType::F32,
                vec![(cfg.dep_q * d) as u64, d as u64],
                f32_bytes(&out_proj),
            )
            .unwrap();
            b.add_tensor(
                &format!("{p}.norm1.alpha"),
                GgmlType::F32,
                vec![1, 1, d as u64],
                f32_bytes(&layer.norm1_gamma),
            )
            .unwrap();
            b.add_tensor(
                &format!("{p}.norm2.alpha"),
                GgmlType::F32,
                vec![1, 1, d as u64],
                f32_bytes(&layer.norm2_gamma),
            )
            .unwrap();
            for s in 0..cfg.dep_q {
                let mut lin_in = Vec::with_capacity(2 * h * d);
                lin_in.extend(transpose(&layer.gate_w_t[s], d, h));
                lin_in.extend(transpose(&layer.up_w_t[s], d, h));
                b.add_tensor(
                    &format!("{p}.gating.{s}.linear_in.weight"),
                    GgmlType::F32,
                    vec![(2 * h) as u64, d as u64],
                    f32_bytes(&lin_in),
                )
                .unwrap();
                b.add_tensor(
                    &format!("{p}.gating.{s}.linear_out.weight"),
                    GgmlType::F32,
                    vec![d as u64, h as u64],
                    f32_bytes(&transpose(&layer.down_w_t[s], h, d)),
                )
                .unwrap();
            }
        }
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let loaded = MoshiDepthWeights::from_gguf(&file, &cfg).expect("bind");
        assert!(!loaded.is_synthesized);
        let reloaded = MoshiDepthTransformer::new(cfg.clone(), loaded).unwrap();

        let mut s1 = MoshiDepthState::new(&cfg).unwrap();
        let mut s2 = MoshiDepthState::new(&cfg).unwrap();
        let mut c1 = vec![0u32; cfg.dep_q];
        let mut c2 = vec![0u32; cfg.dep_q];
        src.decode_frame(&hidden(&cfg, 4), 2, &mut s1, &mut c1, |l| argmax(l))
            .unwrap();
        reloaded
            .decode_frame(&hidden(&cfg, 4), 2, &mut s2, &mut c2, |l| argmax(l))
            .unwrap();
        assert_eq!(c1, c2, "pack → GGUF → unpack is exact");
    }
}
