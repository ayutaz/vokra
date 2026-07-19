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
    CausalConv1d, ConvState, MimiTransformer, MimiTransformerLayer, MimiTransformerState, PadMode,
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
    /// `[q_dim, dim]` row-major (GEMV layout) — quantizer input projection
    /// (the **semantic** split's `rvq_first.input_proj` when
    /// `input_proj_rest` is present, else the single-chain projection).
    input_proj: Vec<f32>,
    /// `[q_dim, dim]` acoustic-split projection
    /// (`quantizer.rvq_rest.input_proj`, upstream
    /// `SplitResidualVectorQuantizer`). `None` → plain single-chain RVQ
    /// (synthesized fixtures / pre-split GGUFs); `Some` → split encode:
    /// codebook 0 quantizes `input_proj(x)` (1-deep chain), codebooks 1..
    /// chain over `input_proj_rest(x)` — **both project the same latent**;
    /// the acoustic chain does *not* consume the semantic residual
    /// (`vq.py` `SplitResidualVectorQuantizer.encode`, transcribed).
    input_proj_rest: Option<Vec<f32>>,
    tables: Vec<CodebookTable>,
    /// Per-codebook transposed data `[d_model, codebook_size]` (dimension-
    /// major, entry fastest), built once at construction (M5-14 Wave-2 T19):
    /// the RVQ nearest-row search sweeps entry-contiguous rows so the
    /// per-entry distance chains auto-vectorize — bit-identical to the
    /// row-major scan (see [`rvq_quantize_chain_t`]).
    tables_t: Vec<Vec<f32>>,
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
    /// `[frames_cap, q_dim]` batch residual buffers for the codebook-outer
    /// RVQ sweep (M5-14 Wave-2 T19).
    proj_all: Vec<f32>,
    rest_all: Vec<f32>,
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
        // Frame resample: causal conv, k = 2·stride, bias-less, replicate
        // left pad (resample.py `ConvDownsample1d(pad_mode="replicate")`
        // — ADR §D2; the SEANet convs stay constant-pad).
        let frame_down =
            conv(&mut rng, dim, dim, 2 * ds, ds, 1, false)?.with_pad_mode(PadMode::Replicate);

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
        let tables_t = transpose_tables(&tables);
        Ok(Self {
            config: config.clone(),
            init,
            stages,
            final_conv,
            frame_down,
            transformer,
            input_proj,
            input_proj_rest: None,
            tables,
            tables_t,
            backend: BackendKind::Cpu,
            is_synthesized: true,
        })
    }

    /// Binds neural-chain weights from a GGUF under the Vokra **structural**
    /// naming (`mimi.enc.*`), reproducing exactly the geometry
    /// [`Self::synthesized`] builds.
    ///
    /// # Naming (T29 — closed by the converter adapter)
    ///
    /// The real kyutai checkpoint
    /// (`tokenizer-e351c8d8-checkpoint125.safetensors`) stores the chain
    /// under `encoder.model.{i}.conv.conv.*` / `encoder_transformer.*` /
    /// `quantizer.*` names with **plain fused conv weights** (loaders.py:
    /// weights are "pre-processed for inference", `norm: "none"` — no
    /// `weight_norm` `(g, v)` split survives in the file) and bias-less
    /// transformer linears. `vokra-convert::models::mimi` maps those names
    /// onto this structural naming offline (index walk + linear transposes
    /// + raw-codebook derivation); this binder stays checkpoint-agnostic.
    ///
    /// The optional `mimi.enc.input_proj_rest` tensor (presence-driven, no
    /// hidden flag) selects the upstream `SplitResidualVectorQuantizer`
    /// encode: codebook 0 = 1-deep semantic chain over `input_proj(x)`,
    /// codebooks 1.. = acoustic chain over `input_proj_rest(x)` (both from
    /// the same latent — `vq.py`, transcribed). Absent → plain single-chain
    /// RVQ (synthesized fixtures / pre-split GGUFs).
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming any missing / mis-shaped tensor
    /// (FR-EX-08); propagates config validation.
    pub fn from_gguf(file: &GgufFile, config: &MimiNeuralConfig) -> Result<Self> {
        config.validate()?;
        let s = &config.seanet;
        let nf = s.n_filters;
        let dim = s.dimension;
        let k = s.kernel_size;

        let init = read_conv(file, "mimi.enc.init", 1, nf, k, 1, 1, true)?;
        let mut ch = nf;
        let mut stages = Vec::with_capacity(s.ratios.len());
        // Same order as `synthesized`: ratios reversed (seanet.py).
        for (i, &r) in s.ratios.iter().rev().enumerate() {
            let hidden = (ch / s.compress).max(1);
            let mut blocks = Vec::with_capacity(s.n_residual_layers);
            for j in 0..s.n_residual_layers {
                let dil = s.dilation_base.pow(j as u32);
                blocks.push(ResBlock {
                    conv1: read_conv(
                        file,
                        &format!("mimi.enc.s{i}.b{j}.c1"),
                        ch,
                        hidden,
                        s.residual_kernel_size,
                        1,
                        dil,
                        true,
                    )?,
                    conv2: read_conv(
                        file,
                        &format!("mimi.enc.s{i}.b{j}.c2"),
                        hidden,
                        ch,
                        1,
                        1,
                        1,
                        true,
                    )?,
                });
            }
            let down = read_conv(
                file,
                &format!("mimi.enc.s{i}.down"),
                ch,
                ch * 2,
                2 * r,
                r,
                1,
                true,
            )?;
            stages.push(EncStage { blocks, down });
            ch *= 2;
        }
        let final_conv = read_conv(
            file,
            "mimi.enc.final",
            ch,
            dim,
            s.last_kernel_size,
            1,
            1,
            true,
        )?;
        let ds = config.frame_downsample_stride()?;
        // Replicate left pad — resample.py `ConvDownsample1d`
        // (`pad_mode="replicate"`); mirrors `synthesized`.
        let frame_down = read_conv(file, "mimi.enc.frame_down", dim, dim, 2 * ds, ds, 1, false)?
            .with_pad_mode(PadMode::Replicate);

        let t = &config.transformer;
        let mut layers = Vec::with_capacity(t.n_layer);
        for l in 0..t.n_layer {
            layers.push(read_tf_layer(
                file,
                &format!("mimi.enc.tf{l}"),
                t.d_model,
                t.ff_dim,
            )?);
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
        let input_proj = tensor_f32(file, "mimi.enc.input_proj", q.dimension * dim)?;
        let input_proj_rest = if file.tensor_info("mimi.enc.input_proj_rest").is_some() {
            Some(tensor_f32(
                file,
                "mimi.enc.input_proj_rest",
                q.dimension * dim,
            )?)
        } else {
            None
        };
        let mut tables = Vec::with_capacity(q.n_q);
        for cb in 0..q.n_q {
            let data = tensor_f32(file, &format!("mimi.enc.cb{cb}"), q.bins * q.dimension)?;
            tables.push(CodebookTable::new(q.bins, q.dimension, data)?);
        }
        let tables_t = transpose_tables(&tables);
        Ok(Self {
            config: config.clone(),
            init,
            stages,
            final_conv,
            frame_down,
            transformer,
            input_proj,
            input_proj_rest,
            tables,
            tables_t,
            backend: BackendKind::Cpu,
            is_synthesized: false,
        })
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

    /// `true` when the upstream `SplitResidualVectorQuantizer` encode is
    /// active (`mimi.enc.input_proj_rest` present — [`Self::from_gguf`]).
    #[must_use]
    pub fn has_split_projection(&self) -> bool {
        self.input_proj_rest.is_some()
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
            proj_all: vec![0.0; frames_cap * self.config.quantizer.dimension],
            rest_all: vec![0.0; frames_cap * self.config.quantizer.dimension],
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

        // Input projections per frame (GEMV — unchanged math), then the RVQ
        // residual chains swept CODEBOOK-outer / frame-inner (M5-14 Wave-2
        // T19): frames are independent, so the loop order does not change any
        // per-(frame, codebook) value — but one ~2 MB codebook now stays
        // cache-resident across all frames instead of all 32 codebooks
        // (~64 MB) being re-streamed from DRAM for every frame (the Wave-0
        // ~1.0 s scalar argmin was bandwidth-bound as much as compute-bound).
        let q_dim = self.config.quantizer.dimension;
        let n_qs = self.tables.len();
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
            state.proj_all[f * q_dim..(f + 1) * q_dim].copy_from_slice(&state.proj);
            if let Some(w_rest) = &self.input_proj_rest {
                // Upstream split encode (vq.py `SplitResidualVector-
                // Quantizer.encode`): the semantic split (rvq_first,
                // 1 codebook) and the acoustic split (rvq_rest, the
                // rest) each project the SAME latent through their own
                // input_proj — the acoustic chain starts from
                // `input_proj_rest(x)`, not the semantic residual.
                compute.gemv_f32(
                    q_dim,
                    dim,
                    w_rest,
                    &state.tf_rows[..dim],
                    None,
                    &mut state.residual,
                )?;
                state.rest_all[f * q_dim..(f + 1) * q_dim].copy_from_slice(&state.residual);
            }
        }
        match &self.input_proj_rest {
            None => {
                // Plain chain over every codebook, frame-inner.
                for (cb, (table, table_t)) in self.tables.iter().zip(&self.tables_t).enumerate() {
                    for f in 0..n_frames {
                        let r = &mut state.proj_all[f * q_dim..(f + 1) * q_dim];
                        codes_out[f * n_qs + cb] = rvq_quantize_one_t(table, table_t, r)?;
                    }
                }
            }
            Some(_) => {
                // Semantic split: codebook 0 over `input_proj(x)`.
                for f in 0..n_frames {
                    let r = &mut state.proj_all[f * q_dim..(f + 1) * q_dim];
                    codes_out[f * n_qs] =
                        rvq_quantize_one_t(&self.tables[0], &self.tables_t[0], r)?;
                }
                // Acoustic split: codebooks 1.. chain over `input_proj_rest(x)`.
                for cb in 1..n_qs {
                    let (table, table_t) = (&self.tables[cb], &self.tables_t[cb]);
                    for f in 0..n_frames {
                        let r = &mut state.rest_all[f * q_dim..(f + 1) * q_dim];
                        codes_out[f * n_qs + cb] = rvq_quantize_one_t(table, table_t, r)?;
                    }
                }
            }
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
///
/// Since M5-14 Wave-2 this row-major scan is the **test-side reference
/// oracle** for the transposed production path ([`rvq_quantize_one_t`]).
#[cfg(test)]
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

/// Per-codebook `[codebook_size, d_model]` → `[d_model, codebook_size]`
/// transposes (load-time only; see `MimiEncoder::tables_t`).
fn transpose_tables(tables: &[CodebookTable]) -> Vec<Vec<f32>> {
    tables
        .iter()
        .map(|t| {
            let mut tt = vec![0.0f32; t.data.len()];
            for e in 0..t.codebook_size {
                for j in 0..t.d_model {
                    tt[j * t.codebook_size + e] = t.data[e * t.d_model + j];
                }
            }
            tt
        })
        .collect()
}

/// [`rvq_quantize_chain`] over pre-transposed codebooks — the M5-14 Wave-2
/// (T19) hot path. Wave-0 measured ~1.0 s of the Mimi encode wall in the
/// row-major scalar argmin scan.
///
/// **Bit-identical to [`rvq_quantize_chain`]** by construction: each entry's
/// squared distance is still one accumulator advanced over the feature
/// dimensions `j` in ascending order with the same unfused
/// `(r − v)·(r − v)` chain, and the winner scan still walks entries in
/// ascending order with the same strict `<` (lowest-index tie-break, the
/// T14-pinned upstream rule). Only the loop nesting changes — dimensions
/// outer, a 16-entry block of distance lanes inner over the
/// entry-contiguous `tables_t` rows — which auto-vectorizes. The
/// differential test below pins `==` on codes AND residual, which is what
/// keeps the real-weight roundtrip (codes 100% match) untouched.
#[cfg(test)]
pub(crate) fn rvq_quantize_chain_t(
    tables: &[CodebookTable],
    tables_t: &[Vec<f32>],
    residual: &mut [f32],
    codes_out: &mut [u32],
) -> Result<()> {
    if codes_out.len() != tables.len() || tables_t.len() != tables.len() {
        return Err(VokraError::InvalidArgument(format!(
            "rvq quantize: codes_out len {} / tables_t len {} != n_q {}",
            codes_out.len(),
            tables_t.len(),
            tables.len()
        )));
    }
    for (cb, (table, table_t)) in tables.iter().zip(tables_t).enumerate() {
        codes_out[cb] = rvq_quantize_one_t(table, table_t, residual)?;
    }
    Ok(())
}

/// One codebook of the transposed-scan RVQ: nearest row of `table` to
/// `residual` (same distance chains and `<` lowest-index tie-break as the
/// row-major scan — see [`rvq_quantize_chain_t`]), then subtracts the
/// winner row from `residual` in place. Returns the winner index.
pub(crate) fn rvq_quantize_one_t(
    table: &CodebookTable,
    table_t: &[f32],
    residual: &mut [f32],
) -> Result<u32> {
    if table.d_model != residual.len() || table_t.len() != table.data.len() {
        return Err(VokraError::InvalidArgument(format!(
            "rvq quantize: table d_model {} != residual len {} (or stale transpose)",
            table.d_model,
            residual.len()
        )));
    }
    /// Distance lanes per block. Full blocks run a constant-trip inner loop
    /// (vectorizes cleanly); the ragged tail (only when `codebook_size` is
    /// not a multiple) reuses the same chain at variable width.
    const BLOCK: usize = 32;
    let size = table.codebook_size;
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    let mut dist = [0.0f32; BLOCK];
    let full = size - size % BLOCK;
    let mut e0 = 0usize;
    while e0 < full {
        dist.fill(0.0);
        for (j, &r) in residual.iter().enumerate() {
            let row = &table_t[j * size + e0..j * size + e0 + BLOCK];
            for (d, &v) in dist.iter_mut().zip(row) {
                let diff = r - v;
                *d += diff * diff;
            }
        }
        // Winner scan in ascending entry order (same `<` tie-break as the
        // row-major reference).
        for (lane, &d) in dist.iter().enumerate() {
            if d < best_d {
                best_d = d;
                best = e0 + lane;
            }
        }
        e0 += BLOCK;
    }
    if e0 < size {
        let lanes = size - e0;
        let block = &mut dist[..lanes];
        block.fill(0.0);
        for (j, &r) in residual.iter().enumerate() {
            let row = &table_t[j * size + e0..j * size + e0 + lanes];
            for (d, &v) in block.iter_mut().zip(row) {
                let diff = r - v;
                *d += diff * diff;
            }
        }
        for (lane, &d) in block.iter().enumerate() {
            if d < best_d {
                best_d = d;
                best = e0 + lane;
            }
        }
    }
    let row = &table.data[best * table.d_model..(best + 1) * table.d_model];
    for (r, v) in residual.iter_mut().zip(row.iter()) {
        *r -= *v;
    }
    Ok(best as u32)
}

/// Reads a named tensor as f32, enforcing the element count (loud
/// [`VokraError::ModelLoad`] on absence / size mismatch — FR-EX-08). Shared
/// by the encoder + decoder neural-chain binders (`mimi:`-prefixed errors).
pub(crate) fn tensor_f32(file: &GgufFile, name: &str, want: usize) -> Result<Vec<f32>> {
    let v = file
        .tensor_f32(name)
        .map_err(|e| VokraError::ModelLoad(format!("mimi: tensor `{name}`: {e}")))?;
    if v.len() != want {
        return Err(VokraError::ModelLoad(format!(
            "mimi: tensor `{name}` has {} elements, expected {want}",
            v.len()
        )));
    }
    Ok(v)
}

/// Reads a `[out, in, k]` causal conv (+ optional `{name}.bias`) from the
/// GGUF and builds it (Vokra structural naming — see
/// [`MimiEncoder::from_gguf`] for the honest naming boundary).
#[allow(clippy::too_many_arguments)] // conv geometry mirrors CausalConv1d::new
pub(crate) fn read_conv(
    file: &GgufFile,
    name: &str,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    stride: usize,
    dil: usize,
    bias: bool,
) -> Result<CausalConv1d> {
    let w = tensor_f32(file, &format!("{name}.weight"), out_ch * in_ch * k)?;
    let b = if bias {
        Some(tensor_f32(file, &format!("{name}.bias"), out_ch)?)
    } else {
        None
    };
    CausalConv1d::new(in_ch, out_ch, k, stride, dil, &w, b)
}

/// Reads one bottleneck-transformer layer (verbatim runtime `w_t` layout).
pub(crate) fn read_tf_layer(
    file: &GgufFile,
    prefix: &str,
    d: usize,
    ff: usize,
) -> Result<MimiTransformerLayer> {
    Ok(MimiTransformerLayer {
        ln1_gamma: tensor_f32(file, &format!("{prefix}.ln1_gamma"), d)?,
        ln1_beta: tensor_f32(file, &format!("{prefix}.ln1_beta"), d)?,
        q_w_t: tensor_f32(file, &format!("{prefix}.q"), d * d)?,
        k_w_t: tensor_f32(file, &format!("{prefix}.k"), d * d)?,
        v_w_t: tensor_f32(file, &format!("{prefix}.v"), d * d)?,
        o_w_t: tensor_f32(file, &format!("{prefix}.o"), d * d)?,
        layer_scale_1: tensor_f32(file, &format!("{prefix}.ls1"), d)?,
        ln2_gamma: tensor_f32(file, &format!("{prefix}.ln2_gamma"), d)?,
        ln2_beta: tensor_f32(file, &format!("{prefix}.ln2_beta"), d)?,
        fc1_w_t: tensor_f32(file, &format!("{prefix}.fc1"), d * ff)?,
        fc2_w_t: tensor_f32(file, &format!("{prefix}.fc2"), ff * d)?,
        layer_scale_2: tensor_f32(file, &format!("{prefix}.ls2"), d)?,
    })
}

/// Writes one bottleneck-transformer layer under `{prefix}.*` (round-trip
/// mirror of [`read_tf_layer`]).
#[cfg(test)]
pub(crate) fn write_tf_layer(
    b: &mut vokra_core::gguf::GgufBuilder,
    prefix: &str,
    l: &MimiTransformerLayer,
) {
    use vokra_core::gguf::GgmlType;
    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let add = |b: &mut vokra_core::gguf::GgufBuilder, suffix: &str, v: &[f32]| {
        b.add_tensor(
            &format!("{prefix}.{suffix}"),
            GgmlType::F32,
            vec![v.len() as u64],
            bytes(v),
        )
        .unwrap();
    };
    add(b, "ln1_gamma", &l.ln1_gamma);
    add(b, "ln1_beta", &l.ln1_beta);
    add(b, "q", &l.q_w_t);
    add(b, "k", &l.k_w_t);
    add(b, "v", &l.v_w_t);
    add(b, "o", &l.o_w_t);
    add(b, "ls1", &l.layer_scale_1);
    add(b, "ln2_gamma", &l.ln2_gamma);
    add(b, "ln2_beta", &l.ln2_beta);
    add(b, "fc1", &l.fc1_w_t);
    add(b, "fc2", &l.fc2_w_t);
    add(b, "ls2", &l.layer_scale_2);
}

/// Writes a causal conv's `[out, in, k]` weight (+ bias) under `{name}.*`
/// (round-trip mirror of [`read_conv`]).
#[cfg(test)]
pub(crate) fn write_conv(b: &mut vokra_core::gguf::GgufBuilder, name: &str, c: &CausalConv1d) {
    use vokra_core::gguf::GgmlType;
    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let w = c.weight_oik();
    b.add_tensor(
        &format!("{name}.weight"),
        GgmlType::F32,
        vec![w.len() as u64],
        bytes(&w),
    )
    .unwrap();
    if let Some(bias) = c.bias() {
        b.add_tensor(
            &format!("{name}.bias"),
            GgmlType::F32,
            vec![bias.len() as u64],
            bytes(bias),
        )
        .unwrap();
    }
}

/// Packs **every** encoder tensor under the structural `mimi.enc.*` naming
/// (the exact round-trip mirror of [`MimiEncoder::from_gguf`]) — test
/// support for the engine-level side-car bind tests (`moshi::engine`),
/// which build tiny standalone-Mimi GGUFs without reaching into the
/// encoder's private fields from another module.
#[cfg(test)]
pub(crate) fn pack_encoder_structural(b: &mut vokra_core::gguf::GgufBuilder, src: &MimiEncoder) {
    use vokra_core::gguf::GgmlType;
    let write_vec = |b: &mut vokra_core::gguf::GgufBuilder, name: &str, v: &[f32]| {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        b.add_tensor(name, GgmlType::F32, vec![v.len() as u64], bytes)
            .unwrap();
    };
    write_conv(b, "mimi.enc.init", &src.init);
    for (i, stage) in src.stages.iter().enumerate() {
        for (j, blk) in stage.blocks.iter().enumerate() {
            write_conv(b, &format!("mimi.enc.s{i}.b{j}.c1"), &blk.conv1);
            write_conv(b, &format!("mimi.enc.s{i}.b{j}.c2"), &blk.conv2);
        }
        write_conv(b, &format!("mimi.enc.s{i}.down"), &stage.down);
    }
    write_conv(b, "mimi.enc.final", &src.final_conv);
    write_conv(b, "mimi.enc.frame_down", &src.frame_down);
    for (l, layer) in src.transformer.layers.iter().enumerate() {
        write_tf_layer(b, &format!("mimi.enc.tf{l}"), layer);
    }
    write_vec(b, "mimi.enc.input_proj", &src.input_proj);
    if let Some(rest) = &src.input_proj_rest {
        write_vec(b, "mimi.enc.input_proj_rest", rest);
    }
    for (cb, table) in src.tables.iter().enumerate() {
        write_vec(b, &format!("mimi.enc.cb{cb}"), &table.data);
    }
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

    /// M5-14 Wave-2 bit-identity pin: the transposed-scan quantizer must
    /// reproduce the row-major reference **exactly** — same distances
    /// (same per-entry chains), same `<` lowest-index tie-break, same
    /// residual subtraction — codes AND residual compared with `==`.
    /// Ragged codebook sizes exercise the tail block.
    #[test]
    fn transposed_quantize_bitwise_matches_reference_chain() {
        let mut rng = SplitMix64::new(97);
        for &(bins, d_model, n_q) in &[(64usize, 8usize, 3usize), (37, 5, 2), (33, 4, 1)] {
            let mut tables = Vec::new();
            for _ in 0..n_q {
                let data: Vec<f32> = (0..bins * d_model)
                    .map(|_| rng.next_unit_f32() * 2.0 - 1.0)
                    .collect();
                tables.push(CodebookTable::new(bins, d_model, data).unwrap());
            }
            let tables_t = transpose_tables(&tables);
            let start: Vec<f32> = (0..d_model)
                .map(|_| rng.next_unit_f32() * 2.0 - 1.0)
                .collect();

            let mut want_res = start.clone();
            let mut want_codes = vec![0u32; n_q];
            rvq_quantize_chain(&tables, &mut want_res, &mut want_codes).unwrap();

            let mut got_res = start.clone();
            let mut got_codes = vec![0u32; n_q];
            rvq_quantize_chain_t(&tables, &tables_t, &mut got_res, &mut got_codes).unwrap();

            assert_eq!(got_codes, want_codes, "codes ({bins}x{d_model}x{n_q})");
            assert_eq!(got_res, want_res, "residual ({bins}x{d_model}x{n_q})");
        }
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

    /// Appends a plain 1-D f32 tensor (input_proj / codebook data).
    fn write_vec(b: &mut vokra_core::gguf::GgufBuilder, name: &str, v: &[f32]) {
        use vokra_core::gguf::GgmlType;
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        b.add_tensor(name, GgmlType::F32, vec![v.len() as u64], bytes)
            .unwrap();
    }

    #[test]
    fn from_gguf_binds_structural_named_tensors_round_trip() {
        // Pack every neural-chain weight under the Vokra structural naming
        // (`mimi.enc.*`), then verify `from_gguf` binds it back to an encoder
        // that reproduces the exact encode. Pack ↔ unpack self-consistency;
        // real kyutai-name / weight_norm mapping stays owner (module docs).
        use vokra_core::gguf::GgufBuilder;
        let cfg = MimiNeuralConfig::tiny_for_tests();
        let src = MimiEncoder::synthesized(&cfg, 5).unwrap();

        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "mimi");
        write_conv(&mut b, "mimi.enc.init", &src.init);
        for (i, stage) in src.stages.iter().enumerate() {
            for (j, blk) in stage.blocks.iter().enumerate() {
                write_conv(&mut b, &format!("mimi.enc.s{i}.b{j}.c1"), &blk.conv1);
                write_conv(&mut b, &format!("mimi.enc.s{i}.b{j}.c2"), &blk.conv2);
            }
            write_conv(&mut b, &format!("mimi.enc.s{i}.down"), &stage.down);
        }
        write_conv(&mut b, "mimi.enc.final", &src.final_conv);
        write_conv(&mut b, "mimi.enc.frame_down", &src.frame_down);
        for (l, layer) in src.transformer.layers.iter().enumerate() {
            write_tf_layer(&mut b, &format!("mimi.enc.tf{l}"), layer);
        }
        write_vec(&mut b, "mimi.enc.input_proj", &src.input_proj);
        for (cb, table) in src.tables.iter().enumerate() {
            write_vec(&mut b, &format!("mimi.enc.cb{cb}"), &table.data);
        }

        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let loaded = MimiEncoder::from_gguf(&file, &cfg).expect("bind");
        assert!(!loaded.is_synthesized());

        // Forward equality: identical PCM → identical RVQ codes.
        let hop = src.frame_hop().unwrap();
        let pcm: Vec<f32> = (0..hop * 3)
            .map(|i| ((i as f32) * 0.017).sin() * 0.3)
            .collect();
        let a = src.encode_all(&pcm).unwrap();
        let e = loaded.encode_all(&pcm).unwrap();
        assert_eq!(a, e, "structural pack → GGUF → bind reproduces the encode");
    }

    #[test]
    fn from_gguf_missing_tensor_is_a_loud_model_load_error() {
        // FR-EX-08: never a silent zero-fill — the first missing conv is loud.
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "mimi");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let err = MimiEncoder::from_gguf(&file, &MimiNeuralConfig::tiny_for_tests()).unwrap_err();
        assert!(matches!(err, VokraError::ModelLoad(_)));
        assert!(
            err.to_string().contains("mimi.enc.init"),
            "names the missing tensor: {err}"
        );
    }

    /// Packs `src` under the structural naming, optionally overriding the
    /// first codebook and appending `mimi.enc.input_proj_rest`.
    fn pack_gguf(src: &MimiEncoder, zero_cb0: bool, input_proj_rest: Option<&[f32]>) -> GgufFile {
        use vokra_core::gguf::GgufBuilder;
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "mimi");
        write_conv(&mut b, "mimi.enc.init", &src.init);
        for (i, stage) in src.stages.iter().enumerate() {
            for (j, blk) in stage.blocks.iter().enumerate() {
                write_conv(&mut b, &format!("mimi.enc.s{i}.b{j}.c1"), &blk.conv1);
                write_conv(&mut b, &format!("mimi.enc.s{i}.b{j}.c2"), &blk.conv2);
            }
            write_conv(&mut b, &format!("mimi.enc.s{i}.down"), &stage.down);
        }
        write_conv(&mut b, "mimi.enc.final", &src.final_conv);
        write_conv(&mut b, "mimi.enc.frame_down", &src.frame_down);
        for (l, layer) in src.transformer.layers.iter().enumerate() {
            write_tf_layer(&mut b, &format!("mimi.enc.tf{l}"), layer);
        }
        write_vec(&mut b, "mimi.enc.input_proj", &src.input_proj);
        if let Some(rest) = input_proj_rest {
            write_vec(&mut b, "mimi.enc.input_proj_rest", rest);
        }
        for (cb, table) in src.tables.iter().enumerate() {
            if cb == 0 && zero_cb0 {
                write_vec(&mut b, "mimi.enc.cb0", &vec![0.0f32; table.data.len()]);
            } else {
                write_vec(&mut b, &format!("mimi.enc.cb{cb}"), &table.data);
            }
        }
        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn split_projection_encode_matches_upstream_split_semantics() {
        // Upstream `SplitResidualVectorQuantizer.encode` (vq.py): semantic
        // (1 codebook on `P(x)`) and acoustic (rest, on `R(x)`) chains both
        // project the SAME latent; the acoustic chain does NOT consume the
        // semantic residual. Oracle: with codebook 0 zeroed, the semantic
        // stage picks index 0 (all-equal distances tie → lowest) and
        // subtracts nothing, so a plain single chain over `P(x)` and the
        // split encode with `R == P` must emit IDENTICAL codes — while a
        // split encode with `R == -P` must diverge on the acoustic stages.
        let cfg = MimiNeuralConfig::tiny_for_tests();
        let src = MimiEncoder::synthesized(&cfg, 5).unwrap();
        let neg: Vec<f32> = src.input_proj.iter().map(|v| -v).collect();

        let plain = MimiEncoder::from_gguf(&pack_gguf(&src, true, None), &cfg).unwrap();
        let split_same =
            MimiEncoder::from_gguf(&pack_gguf(&src, true, Some(&src.input_proj)), &cfg).unwrap();
        let split_neg = MimiEncoder::from_gguf(&pack_gguf(&src, true, Some(&neg)), &cfg).unwrap();
        assert!(!plain.has_split_projection());
        assert!(split_same.has_split_projection());

        let hop = src.frame_hop().unwrap();
        let pcm: Vec<f32> = (0..hop * 4)
            .map(|i| ((i as f32) * 0.023).sin() * 0.4)
            .collect();
        let a = plain.encode_all(&pcm).unwrap();
        let b = split_same.encode_all(&pcm).unwrap();
        let c = split_neg.encode_all(&pcm).unwrap();
        assert_eq!(
            a, b,
            "with cb0 = 0 and R == P the split encode must reduce to the plain chain"
        );
        let n_q = cfg.quantizer.n_q;
        for f in 0..4 {
            assert_eq!(c[f * n_q], b[f * n_q], "semantic stage ignores R");
        }
        assert_ne!(c, b, "acoustic stages must consume input_proj_rest");
    }

    #[test]
    fn split_projection_round_trips_through_gguf() {
        // Presence-driven binding: same GGUF minus the rest tensor loads
        // as a plain encoder; with it, the split path — and the split
        // encode is deterministic across two loads.
        let cfg = MimiNeuralConfig::tiny_for_tests();
        let src = MimiEncoder::synthesized(&cfg, 11).unwrap();
        let rest: Vec<f32> = src.input_proj.iter().map(|v| v * 0.5 + 0.01).collect();
        let file = pack_gguf(&src, false, Some(&rest));
        let e1 = MimiEncoder::from_gguf(&file, &cfg).unwrap();
        let e2 = MimiEncoder::from_gguf(&file, &cfg).unwrap();
        let hop = src.frame_hop().unwrap();
        let pcm: Vec<f32> = (0..hop * 3)
            .map(|i| ((i as f32) * 0.017).cos() * 0.3)
            .collect();
        assert_eq!(e1.encode_all(&pcm).unwrap(), e2.encode_all(&pcm).unwrap());
    }
}
