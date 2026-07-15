//! CSM depth transformer — codebook-axis autoregression conditioned on the
//! backbone hidden state (M4-05-T09).
//!
//! # Terminology (ADR M4-05 §D1-(e))
//!
//! "Depth transformer" is the FR-MD-08 internal term; the upstream
//! `SesameAILabs/csm` `models.py` calls this stack `decoder`
//! (`decoder_flavor="llama-100M"`). The rustdoc sticks to the internal term.
//!
//! # Per-frame autoregression (ADR §D2, transcribed from `generate_frame`)
//!
//! For each 12.5 Hz frame the depth transformer runs a **short, per-frame**
//! sequence over the codebook axis and is reset afterwards
//! (`decoder.reset_caches()` per frame upstream):
//!
//! - position 0: `projection(backbone_hidden)`;
//! - position 1: `projection(audio_embedding(0, c0))` → logits via
//!   `audio_head[0]` → sample `c1`;
//! - position `i` (2 ≤ i < n_codebooks):
//!   `projection(audio_embedding(i-1, c_{i-1}))` → logits via
//!   `audio_head[i-1]` → sample `c_i`.
//!
//! The audio embeddings are the **backbone's** table
//! ([`super::backbone::CsmBackbone::audio_embedding`]) projected to the
//! depth width — the depth transformer owns no embedding table of its own
//! (`models.py`: `self.projection(curr_h)` over backbone-dim rows).
//!
//! # KV: deliberately not paged (ADR §D4)
//!
//! The per-frame KV holds at most `n_codebooks` positions and resets every
//! frame; page management would cost more than it saves. A fixed
//! pre-allocated scratch ([`CsmDepthState`]) carries it with zero
//! allocation in the frame loop (FR-EX-05).

use vokra_core::rng::SplitMix64;
use vokra_core::{BackendKind, Result, VokraError};

use super::backbone::{CsmBackbone, xavier_uniform};
use super::config::CsmConfig;
use super::rope::{llama3_inv_freqs, rope_apply_adjacent};
use crate::compute::{Compute, HotOp};
use crate::cosyvoice2::llm::LlmBlockWeights;
use crate::voxtral::text_decoder::{hadamard_inplace, rms_norm, silu_inplace};

/// Depth-transformer hot ops (same set as the backbone — one coverage gate
/// for the whole CSM step).
const DEPTH_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::Gemv, HotOp::Softmax];

/// All depth-transformer weights (ADR §D2).
///
/// Layouts:
/// - `projection_w_t` — `[d_backbone, d_depth]` row-major, **already
///   transposed** for `Compute::gemm_f32` (upstream `projection` is
///   `Linear(backbone_dim → decoder_dim, bias=False)`);
/// - `blocks` — Llama block bundles at depth dims (shared bundle type with
///   CosyVoice2/backbone — one GQA fix lands for all);
/// - `final_norm_gamma` — `[d_depth]`;
/// - `audio_head` — `[(n_codebooks - 1) * audio_vocab * d_depth]`; slice
///   `i-1` is `[audio_vocab, d_depth]` row-major (**GEMV layout** — the
///   upstream parameter is `[n_codebooks-1, d_depth, audio_vocab]` and is
///   transposed once at load time, T29 converter note).
#[derive(Debug, Clone)]
pub struct CsmDepthWeights {
    /// Backbone→depth projection `[d_backbone, d_depth]` (transposed).
    pub projection_w_t: Vec<f32>,
    /// Per-layer depth transformer blocks.
    pub blocks: Vec<LlmBlockWeights>,
    /// Final RMSNorm γ `[d_depth]`.
    pub final_norm_gamma: Vec<f32>,
    /// Per-codebook heads, slice `i-1 = [audio_vocab, d_depth]` row-major.
    pub audio_head: Vec<f32>,
    /// `true` when built by [`Self::synthesized`].
    pub is_synthesized: bool,
}

impl CsmDepthWeights {
    /// Synthesized (seed-deterministic) store — Xavier projections, γ = 1.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on an ill-formed config.
    pub fn synthesized(config: &CsmConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let d_b = config.backbone.d_model;
        let d = config.depth.d_model;
        let kv_hidden = config.depth.kv_hidden_dim();
        let ffn = config.depth.ffn_dim;
        let n_heads = config.n_codebooks - 1;
        let vocab = config.audio_vocab_size;
        let mut rng = SplitMix64::new(seed);
        let projection_w_t = xavier_uniform(&mut rng, d_b * d, d_b, d);
        let mut blocks = Vec::with_capacity(config.depth.n_layer);
        for _ in 0..config.depth.n_layer {
            blocks.push(LlmBlockWeights {
                attn_norm_gamma: vec![1.0f32; d],
                q_w_t: xavier_uniform(&mut rng, d * d, d, d),
                k_w_t: xavier_uniform(&mut rng, d * kv_hidden, d, kv_hidden),
                v_w_t: xavier_uniform(&mut rng, d * kv_hidden, d, kv_hidden),
                o_w_t: xavier_uniform(&mut rng, d * d, d, d),
                ffn_norm_gamma: vec![1.0f32; d],
                ffn_gate_w_t: xavier_uniform(&mut rng, d * ffn, d, ffn),
                ffn_up_w_t: xavier_uniform(&mut rng, d * ffn, d, ffn),
                ffn_down_w_t: xavier_uniform(&mut rng, ffn * d, ffn, d),
            });
        }
        let final_norm_gamma = vec![1.0f32; d];
        let audio_head = xavier_uniform(&mut rng, n_heads * vocab * d, d, vocab);
        Ok(Self {
            projection_w_t,
            blocks,
            final_norm_gamma,
            audio_head,
            is_synthesized: true,
        })
    }

    /// Real-weight binding — **honest stub** until the T29 tensor manifest
    /// (never a silent zero-fill, FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::NotImplemented`].
    pub fn from_gguf(_file: &vokra_core::gguf::GgufFile, _config: &CsmConfig) -> Result<Self> {
        Err(VokraError::NotImplemented(
            "CSM depth-transformer real-weight binding is deferred to the T29 \
             checkpoint hand-off (ADR M4-05 §D2). Use CsmDepthWeights::synthesized \
             for the deterministic fixture path.",
        ))
    }
}

/// Per-frame depth decode state: fixed-capacity per-layer K/V scratch
/// (`n_codebooks` positions max), reset every frame with **zero
/// allocation** (`begin_frame` only rewinds the position counter).
pub struct CsmDepthState {
    seq_len: usize,
    /// Per-layer K cache `[n_layer][n_codebooks * kv_hidden]` flattened.
    k_cache: Vec<f32>,
    /// Per-layer V cache, same layout.
    v_cache: Vec<f32>,
    // Step scratch (t = 1).
    h: Vec<f32>,
    norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    rope_buf: Vec<f32>,
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

impl std::fmt::Debug for CsmDepthState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsmDepthState")
            .field("seq_len", &self.seq_len)
            .finish()
    }
}

impl CsmDepthState {
    /// Pre-allocates the per-frame scratch for `config`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on an ill-formed config.
    pub fn new(config: &CsmConfig) -> Result<Self> {
        config.validate_for_forward()?;
        let d = config.depth.d_model;
        let kv_hidden = config.depth.kv_hidden_dim();
        let head_dim = config.depth.head_dim();
        let ffn = config.depth.ffn_dim;
        let max_pos = config.n_codebooks; // position 0 + (n_codebooks - 1) code steps
        let n_layer = config.depth.n_layer;
        Ok(Self {
            seq_len: 0,
            k_cache: vec![0.0; n_layer * max_pos * kv_hidden],
            v_cache: vec![0.0; n_layer * max_pos * kv_hidden],
            h: vec![0.0; d],
            norm: vec![0.0; d],
            q: vec![0.0; d],
            k: vec![0.0; kv_hidden],
            v: vec![0.0; kv_hidden],
            rope_buf: vec![0.0; head_dim],
            scores: vec![0.0; max_pos],
            probs: vec![0.0; max_pos],
            attn_out: vec![0.0; d],
            attn_o: vec![0.0; d],
            ffn_gate: vec![0.0; ffn],
            ffn_up: vec![0.0; ffn],
            ffn_down: vec![0.0; d],
            proj_in: vec![0.0; d],
            logits: vec![0.0; config.audio_vocab_size],
        })
    }

    /// Rewinds to an empty frame (upstream `decoder.reset_caches()` per
    /// frame). No allocation — the scratch is retained.
    pub fn begin_frame(&mut self) {
        self.seq_len = 0;
    }

    /// Positions decoded in the current frame.
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
}

/// The depth transformer (config + weights + backend + precomputed RoPE
/// frequencies at depth head width).
pub struct CsmDepthTransformer {
    config: CsmConfig,
    weights: CsmDepthWeights,
    backend: BackendKind,
    inv_freqs: Vec<f32>,
}

impl std::fmt::Debug for CsmDepthTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsmDepthTransformer")
            .field("depth", &self.config.depth)
            .field("weights.is_synthesized", &self.weights.is_synthesized)
            .field("backend", &self.backend)
            .finish()
    }
}

impl CsmDepthTransformer {
    /// Builds the depth transformer from an explicit weight store.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on config/shape mismatch.
    pub fn new(config: CsmConfig, weights: CsmDepthWeights) -> Result<Self> {
        config.validate_for_forward()?;
        validate_depth_shapes(&config, &weights)?;
        let inv_freqs = llama3_inv_freqs(
            config.depth.head_dim(),
            config.rope_base,
            config.rope_scaling.as_ref(),
        )?;
        Ok(Self {
            config,
            weights,
            backend: BackendKind::Cpu,
            inv_freqs,
        })
    }

    /// Synthesized-fixture constructor.
    ///
    /// # Errors
    ///
    /// Propagates [`CsmDepthWeights::synthesized`].
    pub fn synthesized(config: CsmConfig, seed: u64) -> Result<Self> {
        let weights = CsmDepthWeights::synthesized(&config, seed)?;
        Self::new(config, weights)
    }

    /// Selects the compute backend.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// The weight store.
    #[must_use]
    pub fn weights(&self) -> &CsmDepthWeights {
        &self.weights
    }

    fn compute(&self) -> Result<Compute> {
        Compute::for_backend(self.backend, DEPTH_HOT_OPS)
    }

    /// Decodes codebooks `1..n_codebooks` for one frame, conditioned on the
    /// backbone hidden state and the already-sampled zeroth code.
    ///
    /// `codes_out[0]` must already carry `c0`; this fills
    /// `codes_out[1..n_codebooks]`. `sample` receives the `[audio_vocab]`
    /// logits row for codebook `i` and returns the sampled id (the M1
    /// [`vokra_core::Sampler`] closure shape — the frame loop passes
    /// `|logits| sampler.sample(logits)`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on shape mismatches or an
    /// out-of-range sampled id (the sampler must return `< audio_vocab`;
    /// FR-EX-08 — a wrong id would silently corrupt the RVQ stream).
    pub fn decode_frame(
        &self,
        backbone: &CsmBackbone,
        backbone_hidden: &[f32],
        state: &mut CsmDepthState,
        codes_out: &mut [u32],
        mut sample: impl FnMut(&mut [f32]) -> u32,
    ) -> Result<()> {
        let n_cb = self.config.n_codebooks;
        let d_b = self.config.backbone.d_model;
        if backbone_hidden.len() != d_b {
            return Err(VokraError::InvalidArgument(format!(
                "csm depth decode_frame: backbone_hidden len {} != d_backbone {d_b}",
                backbone_hidden.len()
            )));
        }
        if codes_out.len() != n_cb {
            return Err(VokraError::InvalidArgument(format!(
                "csm depth decode_frame: codes_out len {} != n_codebooks {n_cb}",
                codes_out.len()
            )));
        }
        let vocab = self.config.audio_vocab_size as u32;
        if codes_out[0] >= vocab {
            return Err(VokraError::InvalidArgument(format!(
                "csm depth decode_frame: c0 {} >= audio_vocab {vocab}",
                codes_out[0]
            )));
        }
        let compute = self.compute()?;
        state.begin_frame();

        // Position 0: projection(backbone_hidden). No logits are read here
        // (upstream samples c1 only after position 1).
        self.project_into(&compute, backbone_hidden, state)?;
        self.step(&compute, state)?;

        for i in 1..n_cb {
            // Position i: projection(audio_embedding(i-1, c_{i-1})).
            let prev = codes_out[i - 1];
            let emb = backbone.audio_embedding(i - 1, prev)?;
            self.project_into(&compute, emb, state)?;
            self.step(&compute, state)?;
            // Head i-1 GEMV over the final-norm hidden → sample c_i.
            let head = self.audio_head_slice(i - 1);
            let d = self.config.depth.d_model;
            rms_norm(
                &state.h[..d],
                &self.weights.final_norm_gamma,
                self.config.rms_norm_eps,
                1,
                &mut state.norm[..d],
            )?;
            compute.gemv_f32(
                vocab as usize,
                d,
                head,
                &state.norm[..d],
                None,
                &mut state.logits,
            )?;
            let code = sample(&mut state.logits);
            if code >= vocab {
                return Err(VokraError::InvalidArgument(format!(
                    "csm depth decode_frame: sampler returned {code} >= audio_vocab {vocab} \
                     (FR-EX-08 — a wrong RVQ id corrupts the stream silently downstream)"
                )));
            }
            codes_out[i] = code;
        }
        Ok(())
    }

    /// `state.proj_in ← x @ projection_w_t` (backbone width → depth width),
    /// then stages it as the next step input in `state.h`.
    fn project_into(&self, compute: &Compute, x: &[f32], state: &mut CsmDepthState) -> Result<()> {
        let d_b = self.config.backbone.d_model;
        let d = self.config.depth.d_model;
        compute.gemm_f32(
            1,
            d,
            d_b,
            x,
            &self.weights.projection_w_t,
            None,
            &mut state.proj_in[..d],
        )?;
        state.h[..d].copy_from_slice(&state.proj_in[..d]);
        Ok(())
    }

    /// One depth-transformer position over `state.h` (pre-norm Llama block
    /// stack, per-frame flat KV). Leaves the **pre-final-norm** hidden in
    /// `state.h` and advances `state.seq_len`.
    fn step(&self, compute: &Compute, state: &mut CsmDepthState) -> Result<()> {
        let cfg = &self.config.depth;
        let d = cfg.d_model;
        let n_head_q = cfg.n_head_q;
        let n_head_kv = cfg.n_head_kv;
        let head_dim = cfg.head_dim();
        let kv_hidden = cfg.kv_hidden_dim();
        let ffn = cfg.ffn_dim;
        let n_kv_groups = n_head_q / n_head_kv;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let eps = self.config.rms_norm_eps;
        let pos = state.seq_len;
        let max_pos = self.config.n_codebooks;
        if pos >= max_pos {
            return Err(VokraError::InvalidArgument(format!(
                "csm depth step: position {pos} >= per-frame capacity {max_pos} — \
                 begin_frame() was not called between frames?"
            )));
        }
        let t_kv = pos + 1;
        for (layer_idx, block) in self.weights.blocks.iter().enumerate() {
            rms_norm(
                &state.h[..d],
                &block.attn_norm_gamma,
                eps,
                1,
                &mut state.norm[..d],
            )?;
            compute.gemm_f32(1, d, d, &state.norm[..d], &block.q_w_t, None, &mut state.q)?;
            compute.gemm_f32(
                1,
                kv_hidden,
                d,
                &state.norm[..d],
                &block.k_w_t,
                None,
                &mut state.k,
            )?;
            compute.gemm_f32(
                1,
                kv_hidden,
                d,
                &state.norm[..d],
                &block.v_w_t,
                None,
                &mut state.v,
            )?;
            for h_q in 0..n_head_q {
                state.rope_buf[..head_dim]
                    .copy_from_slice(&state.q[h_q * head_dim..(h_q + 1) * head_dim]);
                rope_apply_adjacent(
                    &mut state.rope_buf[..head_dim],
                    1,
                    head_dim,
                    &self.inv_freqs,
                    pos,
                )?;
                state.q[h_q * head_dim..(h_q + 1) * head_dim]
                    .copy_from_slice(&state.rope_buf[..head_dim]);
            }
            for h_kv in 0..n_head_kv {
                state.rope_buf[..head_dim]
                    .copy_from_slice(&state.k[h_kv * head_dim..(h_kv + 1) * head_dim]);
                rope_apply_adjacent(
                    &mut state.rope_buf[..head_dim],
                    1,
                    head_dim,
                    &self.inv_freqs,
                    pos,
                )?;
                state.k[h_kv * head_dim..(h_kv + 1) * head_dim]
                    .copy_from_slice(&state.rope_buf[..head_dim]);
            }
            // Append to the per-frame flat KV (layer-major).
            let layer_base = layer_idx * max_pos * kv_hidden;
            state.k_cache[layer_base + pos * kv_hidden..layer_base + (pos + 1) * kv_hidden]
                .copy_from_slice(&state.k);
            state.v_cache[layer_base + pos * kv_hidden..layer_base + (pos + 1) * kv_hidden]
                .copy_from_slice(&state.v);

            // Single-row causal attention over the frame history.
            for h_q in 0..n_head_q {
                let h_kv = h_q / n_kv_groups;
                let q_row = &state.q[h_q * head_dim..(h_q + 1) * head_dim];
                for j in 0..t_kv {
                    let k_row = &state.k_cache[layer_base + j * kv_hidden + h_kv * head_dim
                        ..layer_base + j * kv_hidden + (h_kv + 1) * head_dim];
                    let mut s = 0.0f32;
                    for c in 0..head_dim {
                        s += q_row[c] * k_row[c];
                    }
                    state.scores[j] = s * scale;
                }
                compute.softmax_f32(&state.scores[..t_kv], &mut state.probs[..t_kv], 1, t_kv)?;
                let out_dst = &mut state.attn_out[h_q * head_dim..(h_q + 1) * head_dim];
                for (c, out) in out_dst.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for j in 0..t_kv {
                        sum += state.probs[j]
                            * state.v_cache[layer_base + j * kv_hidden + h_kv * head_dim + c];
                    }
                    *out = sum;
                }
            }
            compute.gemm_f32(
                1,
                d,
                d,
                &state.attn_out[..d],
                &block.o_w_t,
                None,
                &mut state.attn_o,
            )?;
            for i in 0..d {
                state.h[i] += state.attn_o[i];
            }

            rms_norm(
                &state.h[..d],
                &block.ffn_norm_gamma,
                eps,
                1,
                &mut state.norm[..d],
            )?;
            compute.gemm_f32(
                1,
                ffn,
                d,
                &state.norm[..d],
                &block.ffn_gate_w_t,
                None,
                &mut state.ffn_gate,
            )?;
            compute.gemm_f32(
                1,
                ffn,
                d,
                &state.norm[..d],
                &block.ffn_up_w_t,
                None,
                &mut state.ffn_up,
            )?;
            silu_inplace(&mut state.ffn_gate);
            hadamard_inplace(&mut state.ffn_gate, &state.ffn_up)?;
            compute.gemm_f32(
                1,
                d,
                ffn,
                &state.ffn_gate,
                &block.ffn_down_w_t,
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

    /// `audio_head` slice for codebook head `i` (`0 ≤ i < n_codebooks-1`),
    /// `[audio_vocab, d_depth]` row-major.
    fn audio_head_slice(&self, i: usize) -> &[f32] {
        let per = self.config.audio_vocab_size * self.config.depth.d_model;
        &self.weights.audio_head[i * per..(i + 1) * per]
    }
}

fn validate_depth_shapes(config: &CsmConfig, weights: &CsmDepthWeights) -> Result<()> {
    let d_b = config.backbone.d_model;
    let d = config.depth.d_model;
    let kv_hidden = config.depth.kv_hidden_dim();
    let ffn = config.depth.ffn_dim;
    if config.n_codebooks < 2 {
        return Err(VokraError::InvalidArgument(format!(
            "csm depth: n_codebooks {} < 2 — the depth transformer decodes \
             codebooks 1.. and needs at least one",
            config.n_codebooks
        )));
    }
    let checks = [
        ("projection_w_t", weights.projection_w_t.len(), d_b * d),
        ("final_norm_gamma", weights.final_norm_gamma.len(), d),
        (
            "audio_head",
            weights.audio_head.len(),
            (config.n_codebooks - 1) * config.audio_vocab_size * d,
        ),
    ];
    for (name, got, want) in checks {
        if got != want {
            return Err(VokraError::InvalidArgument(format!(
                "csm CsmDepthTransformer::new: {name} len {got} != expected {want}"
            )));
        }
    }
    if weights.blocks.len() != config.depth.n_layer {
        return Err(VokraError::InvalidArgument(format!(
            "csm CsmDepthTransformer::new: blocks {} != n_layer {}",
            weights.blocks.len(),
            config.depth.n_layer
        )));
    }
    for (i, b) in weights.blocks.iter().enumerate() {
        let checks = [
            ("attn_norm_gamma", b.attn_norm_gamma.len(), d),
            ("q_w_t", b.q_w_t.len(), d * d),
            ("k_w_t", b.k_w_t.len(), d * kv_hidden),
            ("v_w_t", b.v_w_t.len(), d * kv_hidden),
            ("o_w_t", b.o_w_t.len(), d * d),
            ("ffn_norm_gamma", b.ffn_norm_gamma.len(), d),
            ("ffn_gate_w_t", b.ffn_gate_w_t.len(), d * ffn),
            ("ffn_up_w_t", b.ffn_up_w_t.len(), d * ffn),
            ("ffn_down_w_t", b.ffn_down_w_t.len(), ffn * d),
        ];
        for (name, got, want) in checks {
            if got != want {
                return Err(VokraError::InvalidArgument(format!(
                    "csm CsmDepthTransformer::new: block[{i}].{name} len {got} != {want}"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::backbone::{CsmBackboneState, CsmFrame};
    use super::*;

    fn pair() -> (CsmBackbone, CsmDepthTransformer) {
        let cfg = CsmConfig::tiny_for_tests();
        let backbone = CsmBackbone::synthesized(cfg.clone(), 7).expect("backbone");
        let depth = CsmDepthTransformer::synthesized(cfg, 11).expect("depth");
        (backbone, depth)
    }

    fn greedy(logits: &mut [f32]) -> u32 {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best as u32
    }

    #[test]
    fn decode_frame_is_deterministic_and_in_range() {
        let (backbone, depth) = pair();
        let cfg = backbone.config().clone();
        let mut bb_state = CsmBackboneState::new(&cfg).unwrap();
        let hidden = backbone.step(&mut bb_state, &CsmFrame::text(2)).unwrap();
        let mut state = CsmDepthState::new(&cfg).unwrap();
        let mut codes1 = vec![0u32; cfg.n_codebooks];
        codes1[0] = 5;
        depth
            .decode_frame(&backbone, &hidden, &mut state, &mut codes1, greedy)
            .unwrap();
        assert!(
            codes1.iter().all(|&c| (c as usize) < cfg.audio_vocab_size),
            "codes in vocab range"
        );
        // Same inputs → same codes (per-frame reset works, T10 property ii).
        let mut codes2 = vec![0u32; cfg.n_codebooks];
        codes2[0] = 5;
        depth
            .decode_frame(&backbone, &hidden, &mut state, &mut codes2, greedy)
            .unwrap();
        assert_eq!(codes1, codes2, "frame-boundary reset must reproduce");
        assert_eq!(state.seq_len(), cfg.n_codebooks, "positions 0..n_codebooks");
    }

    #[test]
    fn different_c0_changes_downstream_codes_or_not_but_stays_valid() {
        // Not a distribution claim — only that a different conditioning c0
        // still yields in-range, deterministic codes.
        let (backbone, depth) = pair();
        let cfg = backbone.config().clone();
        let mut bb_state = CsmBackboneState::new(&cfg).unwrap();
        let hidden = backbone.step(&mut bb_state, &CsmFrame::text(1)).unwrap();
        let mut state = CsmDepthState::new(&cfg).unwrap();
        let mut a = vec![0u32; cfg.n_codebooks];
        a[0] = 1;
        depth
            .decode_frame(&backbone, &hidden, &mut state, &mut a, greedy)
            .unwrap();
        let mut b = vec![0u32; cfg.n_codebooks];
        b[0] = 1;
        depth
            .decode_frame(&backbone, &hidden, &mut state, &mut b, greedy)
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn shape_and_range_errors_are_loud() {
        let (backbone, depth) = pair();
        let cfg = backbone.config().clone();
        let mut state = CsmDepthState::new(&cfg).unwrap();
        let hidden = vec![0.0f32; cfg.backbone.d_model];
        // Wrong hidden width.
        let mut codes = vec![0u32; cfg.n_codebooks];
        assert!(
            depth
                .decode_frame(&backbone, &hidden[1..], &mut state, &mut codes, greedy)
                .is_err()
        );
        // Wrong codes_out length.
        let mut short = vec![0u32; cfg.n_codebooks - 1];
        assert!(
            depth
                .decode_frame(&backbone, &hidden, &mut state, &mut short, greedy)
                .is_err()
        );
        // Out-of-range c0.
        let mut bad = vec![0u32; cfg.n_codebooks];
        bad[0] = cfg.audio_vocab_size as u32;
        assert!(
            depth
                .decode_frame(&backbone, &hidden, &mut state, &mut bad, greedy)
                .is_err()
        );
        // A sampler that returns an out-of-range id is rejected (FR-EX-08).
        let mut codes = vec![0u32; cfg.n_codebooks];
        let vocab = cfg.audio_vocab_size as u32;
        assert!(matches!(
            depth.decode_frame(&backbone, &hidden, &mut state, &mut codes, |_| vocab),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn wrong_shape_weights_are_rejected() {
        let cfg = CsmConfig::tiny_for_tests();
        let mut w = CsmDepthWeights::synthesized(&cfg, 3).unwrap();
        w.audio_head.pop();
        assert!(matches!(
            CsmDepthTransformer::new(cfg, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn from_gguf_weights_are_an_honest_not_implemented_stub() {
        let cfg = CsmConfig::tiny_for_tests();
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "csm");
        let file = vokra_core::gguf::GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            CsmDepthWeights::from_gguf(&file, &cfg),
            Err(VokraError::NotImplemented(_))
        ));
    }
}
