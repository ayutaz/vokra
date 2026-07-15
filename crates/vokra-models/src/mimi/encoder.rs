//! Mimi encoder — PCM → SEANet downsampling → bottleneck transformer →
//! frame-rate resample → RVQ quantize (M4-05-T12/T13).
//!
//! # Chain (ADR M4-05 §D2; `MimiModel._encode_to_unquantized_latent` order
//! verified upstream)
//!
//! ```text
//! pcm [1, t] ── SEANet encoder ──> latent [dim, t/seanet_hop]   (25 Hz)
//!            ── encoder transformer (in place, 25 Hz)
//!            ── frame resample conv (stride 2)  ──> [dim, frames] (12.5 Hz)
//!            ── input_proj (dim → q_dim) ── RVQ quantize ──> codes [frames, n_q]
//! ```
//!
//! SEANet encoder stage order (seanet.py, transcribed): init conv →
//! per ratio (encoder consumes `ratios` **reversed**): `n_residual_layers`
//! residual blocks (each `[ELU → conv(k=residual_kernel_size,
//! dilation=dilation_base^j) → ELU → conv(k=1)]` with identity skip,
//! `hidden = ch / compress`) → ELU → downsample conv (`k = 2·ratio`,
//! `stride = ratio`, channels ×2) → finally ELU → conv(→ `dimension`,
//! `k = last_kernel_size`).
//!
//! # RVQ quantize (T13 — the decode lookup's inverse)
//!
//! Plain residual chain over the **shared**
//! [`vokra_ops::mimi_rvq::CodebookTable`]s (spec M4-05-T13; ADR §D1-(c) —
//! the table is never held twice): per codebook, nearest row by L2
//! (FP32 accumulation, ties → lowest index — the upstream tie-break is
//! pinned at the T14 real-fixture parity), subtract, next. The
//! `SplitResidualVectorQuantizer` semantic/acoustic split arithmetic is
//! confirmed against real fixtures at T14/T29; a divergence there revises
//! this module + ADR §D7 (honest re-definition, never a fabricated pass).
//!
//! # Streaming
//!
//! Every conv carries its causal left context in [`MimiEncoderState`];
//! feeding one long buffer equals feeding frame-sized chunks bit-for-bit
//! (nn.rs contract). Input must arrive in whole frames
//! (`frame_hop_samples` multiples) — anything else is a loud error
//! (FR-EX-08; the caller buffers partial frames).

use vokra_core::gguf::GgufFile;
use vokra_core::rng::SplitMix64;
use vokra_core::{BackendKind, Result, VokraError};
use vokra_ops::mimi_rvq::CodebookTable;

use super::config::MimiNeuralConfig;
use super::nn::{
    CausalConv1d, ConvState, MimiTransformer, MimiTransformerLayer, MimiTransformerState,
    elu_inplace,
};
use crate::compute::{Compute, HotOp};
use crate::csm::backbone::xavier_uniform;

/// Hot ops the Mimi neural chain dispatches (im2col-GEMM convolutions +
/// transformer GEMM/softmax/LayerNorm/GELU).
pub(crate) const MIMI_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
];

/// One SEANet residual block (assembled).
#[derive(Debug, Clone)]
pub(crate) struct ResBlock {
    pub(crate) conv1: CausalConv1d,
    pub(crate) conv2: CausalConv1d,
}

/// One encoder stage: residual blocks + downsample conv.
#[derive(Debug, Clone)]
pub(crate) struct EncStage {
    pub(crate) blocks: Vec<ResBlock>,
    pub(crate) down: CausalConv1d,
}

/// The Mimi encoder (audio → RVQ codes) — shared component (CSM lands it,
/// M4-06 Moshi consumes; ADR §D1-(c)).
pub struct MimiEncoder {
    config: MimiNeuralConfig,
    init: CausalConv1d,
    stages: Vec<EncStage>,
    final_conv: CausalConv1d,
    frame_down: CausalConv1d,
    transformer: MimiTransformer,
    /// `[q_dim, dim]` row-major (GEMV layout) — quantizer input projection.
    input_proj: Vec<f32>,
    tables: Vec<CodebookTable>,
    backend: BackendKind,
    is_synthesized: bool,
}

impl std::fmt::Debug for MimiEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MimiEncoder")
            .field("config", &self.config)
            .field("is_synthesized", &self.is_synthesized)
            .field("backend", &self.backend)
            .finish()
    }
}

/// All boundary states + pre-allocated pipeline buffers for one
/// [`MimiEncoder`] session (frame-loop path is allocation-free —
/// FR-EX-05).
pub struct MimiEncoderState {
    frames_cap: usize,
    init: ConvState,
    stages: Vec<(Vec<(ConvState, ConvState)>, ConvState)>,
    final_conv: ConvState,
    frame_down: ConvState,
    tf: MimiTransformerState,
    /// Per-step activation buffers, one per pipeline edge (sizes fixed at
    /// construction).
    bufs: Vec<Vec<f32>>,
    /// Residual-skip / ELU scratch per stage width.
    tmp: Vec<Vec<f32>>,
    mid: Vec<Vec<f32>>,
    tf_rows: Vec<f32>,
    proj: Vec<f32>,
    residual: Vec<f32>,
}

impl std::fmt::Debug for MimiEncoderState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MimiEncoderState")
            .field("frames_cap", &self.frames_cap)
            .finish()
    }
}

impl MimiEncoder {
    /// Builds a synthesized (seed-deterministic) encoder for `config` —
    /// shapes and streaming behaviour verified without real weights.
    ///
    /// # Errors
    ///
    /// Propagates config validation errors.
    pub fn synthesized(config: &MimiNeuralConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        let mut rng = SplitMix64::new(seed);
        let s = &config.seanet;
        let nf = s.n_filters;
        let dim = s.dimension;
        let k = s.kernel_size;

        let conv = |rng: &mut SplitMix64,
                    in_ch: usize,
                    out_ch: usize,
                    k: usize,
                    stride: usize,
                    dil: usize,
                    bias: bool|
         -> Result<CausalConv1d> {
            let w = xavier_uniform(rng, out_ch * in_ch * k, in_ch * k, out_ch);
            let b = bias.then(|| xavier_uniform(rng, out_ch, in_ch * k, out_ch));
            CausalConv1d::new(in_ch, out_ch, k, stride, dil, &w, b)
        };

        let init = conv(&mut rng, 1, nf, k, 1, 1, true)?;
        let mut ch = nf;
        let mut stages = Vec::with_capacity(s.ratios.len());
        // Encoder consumes the (decoder-order) ratios reversed (seanet.py).
        for &r in s.ratios.iter().rev() {
            let hidden = (ch / s.compress).max(1);
            let mut blocks = Vec::with_capacity(s.n_residual_layers);
            for j in 0..s.n_residual_layers {
                let dil = s.dilation_base.pow(j as u32);
                blocks.push(ResBlock {
                    conv1: conv(&mut rng, ch, hidden, s.residual_kernel_size, 1, dil, true)?,
                    conv2: conv(&mut rng, hidden, ch, 1, 1, 1, true)?,
                });
            }
            let down = conv(&mut rng, ch, ch * 2, 2 * r, r, 1, true)?;
            stages.push(EncStage { blocks, down });
            ch *= 2;
        }
        let final_conv = conv(&mut rng, ch, dim, s.last_kernel_size, 1, 1, true)?;
        let ds = config.frame_downsample_stride()?;
        // Frame resample: causal conv, k = 2·stride, bias-less
        // (resample.py ConvDownsample1d — ADR §D2).
        let frame_down = conv(&mut rng, dim, dim, 2 * ds, ds, 1, false)?;

        let t = &config.transformer;
        let mut layers = Vec::with_capacity(t.n_layer);
        for _ in 0..t.n_layer {
            layers.push(synthesized_transformer_layer(
                &mut rng,
                t.d_model,
                t.ff_dim,
                t.layer_scale,
            ));
        }
        let transformer = MimiTransformer::new(
            t.d_model,
            t.n_head,
            t.ff_dim,
            t.context,
            t.max_period,
            layers,
        )?;

        let q = &config.quantizer;
        let input_proj = xavier_uniform(&mut rng, q.dimension * dim, dim, q.dimension);
        let mut tables = Vec::with_capacity(q.n_q);
        for _ in 0..q.n_q {
            let data = xavier_uniform(&mut rng, q.bins * q.dimension, q.dimension, q.dimension);
            tables.push(CodebookTable::new(q.bins, q.dimension, data)?);
        }
        Ok(Self {
            config: config.clone(),
            init,
            stages,
            final_conv,
            frame_down,
            transformer,
            input_proj,
            tables,
            backend: BackendKind::Cpu,
            is_synthesized: true,
        })
    }

    /// Real-weight binding — **honest stub** until the T29 kyutai
    /// checkpoint manifest (FR-EX-08, never a zero-fill).
    ///
    /// # Errors
    ///
    /// [`VokraError::NotImplemented`].
    pub fn from_gguf(_file: &GgufFile, _config: &MimiNeuralConfig) -> Result<Self> {
        Err(VokraError::NotImplemented(
            "Mimi encoder real-weight binding is deferred to the T29 checkpoint \
             hand-off (kyutai tensor names — ADR M4-05 §D2). Use \
             MimiEncoder::synthesized for the deterministic fixture path.",
        ))
    }

    /// Selects the Compute-seam backend.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// `true` when built by [`Self::synthesized`].
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.is_synthesized
    }

    /// The resolved config.
    #[must_use]
    pub fn config(&self) -> &MimiNeuralConfig {
        &self.config
    }

    /// The shared RVQ codebook tables (same objects the decode side
    /// consumes — never duplicated, ADR §D1-(c)).
    #[must_use]
    pub fn tables(&self) -> &[CodebookTable] {
        &self.tables
    }

    /// PCM samples per frame.
    ///
    /// # Errors
    ///
    /// Propagates the config rate check.
    pub fn frame_hop(&self) -> Result<usize> {
        self.config.frame_hop_samples()
    }

    fn compute(&self) -> Result<Compute> {
        Compute::for_backend(self.backend, MIMI_HOT_OPS)
    }

    /// Fresh streaming state accepting up to `frames_cap` frames per
    /// `encode_into` call.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on `frames_cap == 0`.
    pub fn state(&self, frames_cap: usize) -> Result<MimiEncoderState> {
        if frames_cap == 0 {
            return Err(VokraError::InvalidArgument(
                "mimi encoder state: frames_cap must be > 0".into(),
            ));
        }
        let hop = self.frame_hop()?;
        let mut t = frames_cap * hop;
        let mut bufs = Vec::new();
        let mut tmp = Vec::new();
        let mut mid = Vec::new();
        // Edge 0: pcm [1, t].
        bufs.push(vec![0.0f32; t]);
        // init conv out [nf, t].
        let mut ch = self.init.out_ch;
        bufs.push(vec![0.0f32; ch * t]);
        let mut stage_states = Vec::new();
        for stage in &self.stages {
            let hidden = stage.blocks.first().map_or(1, |b| b.conv1.out_ch);
            tmp.push(vec![0.0f32; ch * t]);
            mid.push(vec![0.0f32; hidden * t]);
            let mut block_states = Vec::new();
            for b in &stage.blocks {
                block_states.push((b.conv1.state(t), b.conv2.state(t)));
            }
            let down_state = stage.down.state(t);
            t /= stage.down.stride;
            ch = stage.down.out_ch;
            bufs.push(vec![0.0f32; ch * t]);
            stage_states.push((block_states, down_state));
        }
        // final conv out [dim, t_lat].
        let final_state = self.final_conv.state(t);
        let dim = self.final_conv.out_ch;
        bufs.push(vec![0.0f32; dim * t]);
        let tf_rows = vec![0.0f32; t * dim];
        // frame resample out [dim, frames].
        let frame_down_state = self.frame_down.state(t);
        let frames = t / self.frame_down.stride;
        bufs.push(vec![0.0f32; dim * frames]);
        Ok(MimiEncoderState {
            frames_cap,
            init: self.init.state(frames_cap * hop),
            stages: stage_states,
            final_conv: final_state,
            frame_down: frame_down_state,
            tf: self.transformer.state(),
            bufs,
            tmp,
            mid,
            tf_rows,
            proj: vec![0.0; self.config.quantizer.dimension],
            residual: vec![0.0; self.config.quantizer.dimension],
        })
    }

    /// Encodes whole frames of PCM into RVQ codes
    /// (`codes_out = [n_frames, n_q]` row-major), carrying all causal
    /// state in `state`. Allocation-free.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a non-whole-frame `pcm` length,
    /// capacity overflow, or shape mismatch.
    pub fn encode_into(
        &self,
        state: &mut MimiEncoderState,
        pcm: &[f32],
        codes_out: &mut [u32],
    ) -> Result<()> {
        let hop = self.frame_hop()?;
        if pcm.is_empty() || pcm.len() % hop != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "mimi encode: pcm len {} is not a positive multiple of the frame hop \
                 {hop} (buffer whole frames — FR-EX-08, no silent zero-pad)",
                pcm.len()
            )));
        }
        let n_frames = pcm.len() / hop;
        if n_frames > state.frames_cap {
            return Err(VokraError::InvalidArgument(format!(
                "mimi encode: {n_frames} frames > state capacity {}",
                state.frames_cap
            )));
        }
        let n_q = self.config.quantizer.n_q;
        if codes_out.len() != n_frames * n_q {
            return Err(VokraError::InvalidArgument(format!(
                "mimi encode: codes_out len {} != n_frames*n_q {}",
                codes_out.len(),
                n_frames * n_q
            )));
        }
        let compute = self.compute()?;
        let mut t = n_frames * hop;
        state.bufs[0][..t].copy_from_slice(pcm);

        // init conv.
        let mut edge = 1;
        {
            let (before, after) = state.bufs.split_at_mut(edge);
            self.init.process_into(
                &compute,
                &mut state.init,
                &before[edge - 1][..t],
                t,
                &mut after[0][..self.init.out_ch * t],
            )?;
        }
        let mut ch = self.init.out_ch;

        // Stages.
        for (si, stage) in self.stages.iter().enumerate() {
            let (block_states, down_state) = &mut state.stages[si];
            // Residual blocks operate in place on bufs[edge].
            for (bi, block) in stage.blocks.iter().enumerate() {
                let hidden = block.conv1.out_ch;
                let x = &mut state.bufs[edge];
                // tmp = elu(x)
                state.tmp[si][..ch * t].copy_from_slice(&x[..ch * t]);
                elu_inplace(&mut state.tmp[si][..ch * t]);
                block.conv1.process_into(
                    &compute,
                    &mut block_states[bi].0,
                    &state.tmp[si][..ch * t],
                    t,
                    &mut state.mid[si][..hidden * t],
                )?;
                elu_inplace(&mut state.mid[si][..hidden * t]);
                // conv2 (k=1) back to ch — write into tmp then add skip.
                block.conv2.process_into(
                    &compute,
                    &mut block_states[bi].1,
                    &state.mid[si][..hidden * t],
                    t,
                    &mut state.tmp[si][..ch * t],
                )?;
                let x = &mut state.bufs[edge];
                for (dst, src) in x[..ch * t].iter_mut().zip(state.tmp[si][..ch * t].iter()) {
                    *dst += *src;
                }
            }
            // ELU → downsample conv into the next edge buffer.
            state.tmp[si][..ch * t].copy_from_slice(&state.bufs[edge][..ch * t]);
            elu_inplace(&mut state.tmp[si][..ch * t]);
            let t_out = t / stage.down.stride;
            let out_ch = stage.down.out_ch;
            stage.down.process_into(
                &compute,
                down_state,
                &state.tmp[si][..ch * t],
                t,
                &mut state.bufs[edge + 1][..out_ch * t_out],
            )?;
            edge += 1;
            ch = out_ch;
            t = t_out;
        }

        // ELU → final conv → latent [dim, t].
        {
            // The last stage's tmp buffer is wide enough (ch*t of the last
            // stage ≥ current ch*t after its own downsample? No — reuse the
            // final edge buffer copy instead).
            let x = &mut state.bufs[edge];
            elu_inplace(&mut x[..ch * t]);
        }
        let dim = self.final_conv.out_ch;
        {
            let (before, after) = state.bufs.split_at_mut(edge + 1);
            self.final_conv.process_into(
                &compute,
                &mut state.final_conv,
                &before[edge][..ch * t],
                t,
                &mut after[0][..dim * t],
            )?;
        }
        edge += 1;

        // Bottleneck transformer at 25 Hz: channel-major → rows, in place,
        // back.
        for i in 0..t {
            for c in 0..dim {
                state.tf_rows[i * dim + c] = state.bufs[edge][c * t + i];
            }
        }
        self.transformer.process_inplace(
            &compute,
            &mut state.tf,
            &mut state.tf_rows[..t * dim],
            t,
        )?;
        for i in 0..t {
            for c in 0..dim {
                state.bufs[edge][c * t + i] = state.tf_rows[i * dim + c];
            }
        }

        // Frame resample (25 → 12.5 Hz).
        let ds = self.frame_down.stride;
        let frames = t / ds;
        debug_assert_eq!(frames, n_frames);
        {
            let (before, after) = state.bufs.split_at_mut(edge + 1);
            self.frame_down.process_into(
                &compute,
                &mut state.frame_down,
                &before[edge][..dim * t],
                t,
                &mut after[0][..dim * frames],
            )?;
        }
        edge += 1;

        // Per frame: input_proj GEMV + RVQ residual chain.
        let q_dim = self.config.quantizer.dimension;
        for f in 0..n_frames {
            // Column f of [dim, frames] gathered into tf_rows[..dim].
            for c in 0..dim {
                state.tf_rows[c] = state.bufs[edge][c * frames + f];
            }
            compute.gemv_f32(
                q_dim,
                dim,
                &self.input_proj,
                &state.tf_rows[..dim],
                None,
                &mut state.proj,
            )?;
            state.residual.copy_from_slice(&state.proj);
            let codes = &mut codes_out[f * self.tables.len()..(f + 1) * self.tables.len()];
            rvq_quantize_chain(&self.tables, &mut state.residual, codes)?;
        }
        Ok(())
    }

    /// Convenience: fresh-state whole-buffer encode returning
    /// `[n_frames, n_q]` codes.
    ///
    /// # Errors
    ///
    /// See [`Self::encode_into`].
    pub fn encode_all(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        let hop = self.frame_hop()?;
        if pcm.is_empty() || pcm.len() % hop != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "mimi encode: pcm len {} is not a positive multiple of the frame hop {hop}",
                pcm.len()
            )));
        }
        let n_frames = pcm.len() / hop;
        let mut state = self.state(n_frames)?;
        let mut codes = vec![0u32; n_frames * self.config.quantizer.n_q];
        self.encode_into(&mut state, pcm, &mut codes)?;
        Ok(codes)
    }
}

/// Plain RVQ residual quantize chain over shared tables (T13): nearest
/// row by L2 (FP32), subtract, next codebook. Ties break to the lowest
/// index (upstream tie-break pinned at T14 parity).
pub(crate) fn rvq_quantize_chain(
    tables: &[CodebookTable],
    residual: &mut [f32],
    codes_out: &mut [u32],
) -> Result<()> {
    if codes_out.len() != tables.len() {
        return Err(VokraError::InvalidArgument(format!(
            "rvq quantize: codes_out len {} != n_q {}",
            codes_out.len(),
            tables.len()
        )));
    }
    for (cb, table) in tables.iter().enumerate() {
        if table.d_model != residual.len() {
            return Err(VokraError::InvalidArgument(format!(
                "rvq quantize: table[{cb}] d_model {} != residual len {}",
                table.d_model,
                residual.len()
            )));
        }
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for row_idx in 0..table.codebook_size {
            let row = &table.data[row_idx * table.d_model..(row_idx + 1) * table.d_model];
            let mut d = 0.0f32;
            for (r, v) in residual.iter().zip(row.iter()) {
                let diff = r - v;
                d += diff * diff;
            }
            if d < best_d {
                best_d = d;
                best = row_idx;
            }
        }
        codes_out[cb] = best as u32;
        let row = &table.data[best * table.d_model..(best + 1) * table.d_model];
        for (r, v) in residual.iter_mut().zip(row.iter()) {
            *r -= *v;
        }
    }
    Ok(())
}

pub(crate) fn synthesized_transformer_layer(
    rng: &mut SplitMix64,
    d: usize,
    ff: usize,
    layer_scale: f32,
) -> MimiTransformerLayer {
    MimiTransformerLayer {
        ln1_gamma: vec![1.0; d],
        ln1_beta: vec![0.0; d],
        q_w_t: xavier_uniform(rng, d * d, d, d),
        k_w_t: xavier_uniform(rng, d * d, d, d),
        v_w_t: xavier_uniform(rng, d * d, d, d),
        o_w_t: xavier_uniform(rng, d * d, d, d),
        layer_scale_1: vec![layer_scale; d],
        ln2_gamma: vec![1.0; d],
        ln2_beta: vec![0.0; d],
        fc1_w_t: xavier_uniform(rng, d * ff, d, ff),
        fc2_w_t: xavier_uniform(rng, ff * d, ff, d),
        layer_scale_2: vec![layer_scale; d],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_ops::mimi_rvq::{MimiRvqAttrs, mimi_rvq_decode};

    fn encoder() -> MimiEncoder {
        MimiEncoder::synthesized(&MimiNeuralConfig::tiny_for_tests(), 5).expect("encoder")
    }

    fn sine_pcm(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.37).sin() * 0.5).collect()
    }

    #[test]
    fn encode_shapes_and_determinism() {
        let enc = encoder();
        let hop = enc.frame_hop().unwrap();
        let pcm = sine_pcm(hop * 3);
        let a = enc.encode_all(&pcm).unwrap();
        let b = enc.encode_all(&pcm).unwrap();
        assert_eq!(a.len(), 3 * enc.config().quantizer.n_q);
        assert_eq!(a, b, "deterministic");
        assert!(
            a.iter()
                .all(|&c| (c as usize) < enc.config().quantizer.bins)
        );
    }

    #[test]
    fn full_buffer_equals_frame_streaming() {
        // The T12 streaming contract at the encoder level: one 3-frame
        // buffer == three 1-frame calls with carried state.
        let enc = encoder();
        let hop = enc.frame_hop().unwrap();
        let n_q = enc.config().quantizer.n_q;
        let pcm = sine_pcm(hop * 3);
        let full = enc.encode_all(&pcm).unwrap();
        let mut st = enc.state(1).unwrap();
        let mut streamed = Vec::new();
        for f in 0..3 {
            let mut codes = vec![0u32; n_q];
            enc.encode_into(&mut st, &pcm[f * hop..(f + 1) * hop], &mut codes)
                .unwrap();
            streamed.extend_from_slice(&codes);
        }
        assert_eq!(full, streamed, "causal state carry-over must be exact");
    }

    #[test]
    fn non_whole_frame_input_is_loud() {
        let enc = encoder();
        let hop = enc.frame_hop().unwrap();
        assert!(enc.encode_all(&sine_pcm(hop + 1)).is_err());
        assert!(enc.encode_all(&[]).is_err());
        let mut st = enc.state(1).unwrap();
        let mut codes = vec![0u32; enc.config().quantizer.n_q];
        // Two frames into a 1-frame-cap state.
        assert!(
            enc.encode_into(&mut st, &sine_pcm(hop * 2), &mut codes)
                .is_err()
        );
    }

    #[test]
    fn quantize_chain_roundtrips_with_the_decode_op() {
        // Two defining residual-chain properties (with *random* synthetic
        // codebooks a monotone error decrease is NOT guaranteed — that is
        // a trained-codebook property — so we pin the exact algebra
        // instead):
        //
        // 1. **Greedy per-stage optimality**: each chosen row is the L2
        //    argmin over its codebook for the residual entering that stage
        //    (ties → lowest index).
        // 2. **Decomposition identity**: mimi_rvq_decode(codes) + final
        //    residual == target — the encode chain is the decode op's
        //    exact inverse up to the final residual (shared tables,
        //    ADR §D1-(c)).
        let enc = encoder();
        let q_dim = enc.config().quantizer.dimension;
        let n_q = enc.config().quantizer.n_q;
        let attrs = MimiRvqAttrs {
            n_codebooks: n_q,
            codebook_size: enc.config().quantizer.bins,
            d_model: q_dim,
        };
        let target: Vec<f32> = (0..q_dim).map(|i| (i as f32 * 0.71).cos() * 0.3).collect();
        let mut residual = target.clone();
        let mut codes = vec![0u32; n_q];
        // Walk the chain manually to check greedy optimality per stage.
        let mut walk = target.clone();
        for (cb, table) in enc.tables().iter().enumerate() {
            let dist = |row_idx: usize| -> f32 {
                let row = &table.data[row_idx * q_dim..(row_idx + 1) * q_dim];
                walk.iter()
                    .zip(row.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum()
            };
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for r in 0..table.codebook_size {
                let d = dist(r);
                if d < best_d {
                    best_d = d;
                    best = r;
                }
            }
            // The chain must have picked exactly this row.
            let mut chain_res = target.clone();
            let mut chain_codes = vec![0u32; cb + 1];
            rvq_quantize_chain(&enc.tables()[..=cb], &mut chain_res, &mut chain_codes).unwrap();
            assert_eq!(
                chain_codes[cb] as usize, best,
                "stage {cb}: chain must pick the L2 argmin"
            );
            let row = &table.data[best * q_dim..(best + 1) * q_dim];
            for (w, v) in walk.iter_mut().zip(row.iter()) {
                *w -= *v;
            }
        }
        // Full chain + decomposition identity via the shared decode op.
        rvq_quantize_chain(enc.tables(), &mut residual, &mut codes).unwrap();
        let rec = mimi_rvq_decode(&codes, 1, enc.tables(), &attrs).unwrap();
        for (i, ((t, r), q)) in target
            .iter()
            .zip(rec.iter())
            .zip(residual.iter())
            .enumerate()
        {
            assert!(
                (t - (r + q)).abs() < 1e-5,
                "dim {i}: target {t} != decode {r} + residual {q}"
            );
        }
    }

    #[test]
    fn from_gguf_is_an_honest_stub() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "csm");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            MimiEncoder::from_gguf(&file, &MimiNeuralConfig::tiny_for_tests()),
            Err(VokraError::NotImplemented(_))
        ));
    }
}
