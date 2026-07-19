//! Mimi neural decoder — RVQ features → 24 kHz PCM (M4-05-T31/T32/T33).
//!
//! # Chain (ADR M4-05 §D2; `MimiModel.decode` order verified upstream)
//!
//! ```text
//! features [frames, feat] ── (optional feature_proj: q_dim → dim)
//!   ── frame resample conv-transpose (stride 2) ──> [dim, 2·frames] (25 Hz)
//!   ── decoder transformer (in place, 25 Hz)
//!   ── SEANet decoder ──> pcm [1, frames · frame_hop]  (24 kHz)
//! ```
//!
//! SEANet decoder stage order (seanet.py, transcribed): init conv
//! (`dimension → n_filters·2^len(ratios)`, `k = kernel_size`) → per ratio
//! (decoder consumes `ratios` **as given** = coarsest first): ELU →
//! transposed conv (`k = 2·ratio`, `stride = ratio`, channels ÷2, causal
//! right-trim) → `n_residual_layers` residual blocks → finally ELU →
//! conv(`n_filters → channels = 1`, `k = last_kernel_size`).
//!
//! # Feature input — two documented shapes (ADR §D1-(c))
//!
//! `vokra_ops::mimi_rvq::mimi_rvq_decode` sums codebook rows at the
//! table's own width:
//!
//! - **Raw-table path** (`with_feature_proj = true`): tables live at the
//!   quantizer width (256 upstream); the decoder first applies the
//!   quantizer output projection (`q_dim → dimension`). This is the shape
//!   the CSM synthesized pipeline uses (the encoder quantizes at `q_dim`).
//! - **Effective-table path** (`with_feature_proj = false`): the M4-04
//!   standalone Mimi GGUF ships pre-projected tables, so features already
//!   arrive at `dimension` and no projection runs.
//!
//! A wrong feature width is a loud [`VokraError::InvalidArgument`]
//! (FR-EX-08).
//!
//! # Streaming (T33)
//!
//! All conv / conv-transpose / transformer state is carried in
//! [`MimiDecoderState`]; decoding one long feature buffer equals decoding
//! frame-by-frame with the state carried over, bit-for-bit (the nn.rs
//! causal contract — pinned by the `full_buffer_equals_frame_streaming`
//! test). The frame loop is allocation-free (FR-EX-05).

use vokra_core::gguf::GgufFile;
use vokra_core::rng::SplitMix64;
use vokra_core::{BackendKind, Result, VokraError};

use super::config::MimiNeuralConfig;
use super::encoder::{
    MIMI_HOT_OPS, ResBlock, read_conv, read_tf_layer, synthesized_transformer_layer, tensor_f32,
};
use super::nn::{
    CausalConv1d, CausalConvTranspose1d, ConvState, ConvTrState, MimiTransformer,
    MimiTransformerState, elu_inplace,
};
use crate::compute::Compute;
use crate::csm::backbone::xavier_uniform;

/// One decoder stage: transposed upsample conv + residual blocks.
#[derive(Debug, Clone)]
struct DecStage {
    up: CausalConvTranspose1d,
    blocks: Vec<ResBlock>,
}

/// The Mimi neural decoder (features → PCM) — shared component (CSM lands
/// it, M4-06 Moshi consumes; the `cosyvoice2/mimi_bridge.rs` "decoder
/// chain" stub's promised implementation).
pub struct MimiNeuralDecoder {
    config: MimiNeuralConfig,
    /// `[dim, q_dim]` row-major GEMV layout (quantizer output projection)
    /// — `None` on the effective-table path (module docs).
    feature_proj: Option<Vec<f32>>,
    frame_up: CausalConvTranspose1d,
    transformer: MimiTransformer,
    init_conv: CausalConv1d,
    stages: Vec<DecStage>,
    final_conv: CausalConv1d,
    backend: BackendKind,
    is_synthesized: bool,
}

impl std::fmt::Debug for MimiNeuralDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MimiNeuralDecoder")
            .field("config", &self.config)
            .field("feature_proj", &self.feature_proj.is_some())
            .field("is_synthesized", &self.is_synthesized)
            .finish()
    }
}

/// Boundary states + pre-allocated pipeline buffers for one decoder
/// session.
pub struct MimiDecoderState {
    frames_cap: usize,
    frame_up: ConvTrState,
    tf: MimiTransformerState,
    init: ConvState,
    stages: Vec<(ConvTrState, Vec<(ConvState, ConvState)>)>,
    final_conv: ConvState,
    bufs: Vec<Vec<f32>>,
    tmp: Vec<Vec<f32>>,
    mid: Vec<Vec<f32>>,
    tf_rows: Vec<f32>,
    proj: Vec<f32>,
}

impl std::fmt::Debug for MimiDecoderState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MimiDecoderState")
            .field("frames_cap", &self.frames_cap)
            .finish()
    }
}

impl MimiNeuralDecoder {
    /// Builds a synthesized (seed-deterministic) decoder.
    /// `with_feature_proj` selects the raw-table (`true`) or
    /// effective-table (`false`) input shape — module docs.
    ///
    /// # Errors
    ///
    /// Propagates config validation errors.
    pub fn synthesized(
        config: &MimiNeuralConfig,
        seed: u64,
        with_feature_proj: bool,
    ) -> Result<Self> {
        config.validate()?;
        let mut rng = SplitMix64::new(seed);
        let s = &config.seanet;
        let dim = s.dimension;
        let nf = s.n_filters;

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
        let convtr = |rng: &mut SplitMix64,
                      in_ch: usize,
                      out_ch: usize,
                      k: usize,
                      stride: usize,
                      bias: bool|
         -> Result<CausalConvTranspose1d> {
            let w = xavier_uniform(rng, in_ch * out_ch * k, in_ch, out_ch * k);
            let b = bias.then(|| xavier_uniform(rng, out_ch, in_ch, out_ch));
            CausalConvTranspose1d::new(in_ch, out_ch, k, stride, w, b)
        };

        let feature_proj = if with_feature_proj {
            let q_dim = config.quantizer.dimension;
            Some(xavier_uniform(&mut rng, dim * q_dim, q_dim, dim))
        } else {
            None
        };
        let ds = config.frame_downsample_stride()?;
        // Frame resample up: causal conv-transpose, k = 2·stride,
        // bias-less (resample.py ConvTrUpsample1d — ADR §D2).
        let frame_up = convtr(&mut rng, dim, dim, 2 * ds, ds, false)?;

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

        let mut ch = nf * (1 << s.ratios.len());
        let init_conv = conv(&mut rng, dim, ch, s.kernel_size, 1, 1, true)?;
        let mut stages = Vec::with_capacity(s.ratios.len());
        // Decoder consumes ratios as given (coarsest first — seanet.py).
        for &r in &s.ratios {
            let out_ch = ch / 2;
            let up = convtr(&mut rng, ch, out_ch, 2 * r, r, true)?;
            let hidden = (out_ch / s.compress).max(1);
            let mut blocks = Vec::with_capacity(s.n_residual_layers);
            for j in 0..s.n_residual_layers {
                let dil = s.dilation_base.pow(j as u32);
                blocks.push(ResBlock {
                    conv1: conv(
                        &mut rng,
                        out_ch,
                        hidden,
                        s.residual_kernel_size,
                        1,
                        dil,
                        true,
                    )?,
                    conv2: conv(&mut rng, hidden, out_ch, 1, 1, 1, true)?,
                });
            }
            stages.push(DecStage { up, blocks });
            ch = out_ch;
        }
        let final_conv = conv(&mut rng, ch, 1, s.last_kernel_size, 1, 1, true)?;
        Ok(Self {
            config: config.clone(),
            feature_proj,
            frame_up,
            transformer,
            init_conv,
            stages,
            final_conv,
            backend: BackendKind::Cpu,
            is_synthesized: true,
        })
    }

    /// Binds neural-chain weights from a GGUF under the Vokra **structural**
    /// naming (`mimi.dec.*`), reproducing exactly the geometry
    /// [`Self::synthesized`] builds. The quantizer output projection
    /// `mimi.dec.feature_proj` is optional (present → raw-table `q_dim`
    /// input; absent → effective-table `dimension` input) and its presence
    /// selects the mode — no hidden flag.
    ///
    /// # Honest boundary (naming)
    ///
    /// Same as [`super::encoder::MimiEncoder::from_gguf`]: the real kyutai
    /// SEANet decoder is a weight-normed `decoder.model.{i}` `nn.Sequential`
    /// (`moshi/modules/seanet.py`) whose exact indices + `weight_norm`
    /// fusion are checkpoint-exact and not observable without the gated
    /// tokenizer — guessing them is banned. This binder uses an explicit
    /// Vokra structural naming pinned by the round-trip; the `mimi.dec.*` ⇄
    /// real-name adapter + `weight_norm` fusion is the T29 owner step. Real
    /// PCM parity stays an owner task.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming any missing / mis-shaped tensor
    /// (FR-EX-08); propagates config validation.
    pub fn from_gguf(file: &GgufFile, config: &MimiNeuralConfig) -> Result<Self> {
        config.validate()?;
        let s = &config.seanet;
        let dim = s.dimension;
        let nf = s.n_filters;

        let feature_proj = if file.tensor_info("mimi.dec.feature_proj").is_some() {
            let q_dim = config.quantizer.dimension;
            Some(tensor_f32(file, "mimi.dec.feature_proj", dim * q_dim)?)
        } else {
            None
        };
        let ds = config.frame_downsample_stride()?;
        let frame_up = read_convtr(file, "mimi.dec.frame_up", dim, dim, 2 * ds, ds, false)?;

        let t = &config.transformer;
        let mut layers = Vec::with_capacity(t.n_layer);
        for l in 0..t.n_layer {
            layers.push(read_tf_layer(
                file,
                &format!("mimi.dec.tf{l}"),
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

        let mut ch = nf * (1 << s.ratios.len());
        let init_conv = read_conv(file, "mimi.dec.init", dim, ch, s.kernel_size, 1, 1, true)?;
        let mut stages = Vec::with_capacity(s.ratios.len());
        // Same order as `synthesized`: ratios as given (coarsest first).
        for (i, &r) in s.ratios.iter().enumerate() {
            let out_ch = ch / 2;
            let up = read_convtr(
                file,
                &format!("mimi.dec.s{i}.up"),
                ch,
                out_ch,
                2 * r,
                r,
                true,
            )?;
            let hidden = (out_ch / s.compress).max(1);
            let mut blocks = Vec::with_capacity(s.n_residual_layers);
            for j in 0..s.n_residual_layers {
                let dil = s.dilation_base.pow(j as u32);
                blocks.push(ResBlock {
                    conv1: read_conv(
                        file,
                        &format!("mimi.dec.s{i}.b{j}.c1"),
                        out_ch,
                        hidden,
                        s.residual_kernel_size,
                        1,
                        dil,
                        true,
                    )?,
                    conv2: read_conv(
                        file,
                        &format!("mimi.dec.s{i}.b{j}.c2"),
                        hidden,
                        out_ch,
                        1,
                        1,
                        1,
                        true,
                    )?,
                });
            }
            stages.push(DecStage { up, blocks });
            ch = out_ch;
        }
        let final_conv = read_conv(
            file,
            "mimi.dec.final",
            ch,
            1,
            s.last_kernel_size,
            1,
            1,
            true,
        )?;
        Ok(Self {
            config: config.clone(),
            feature_proj,
            frame_up,
            transformer,
            init_conv,
            stages,
            final_conv,
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

    /// The Compute-seam backend the neural decode dispatches through
    /// (default [`BackendKind::Cpu`]).
    #[must_use]
    pub fn backend(&self) -> BackendKind {
        self.backend
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

    /// The per-frame feature width this decoder accepts (`q_dim` on the
    /// raw-table path, `dimension` on the effective-table path).
    #[must_use]
    pub fn expected_feature_dim(&self) -> usize {
        if self.feature_proj.is_some() {
            self.config.quantizer.dimension
        } else {
            self.config.seanet.dimension
        }
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

    /// Fresh streaming state accepting up to `frames_cap` frames per call.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on `frames_cap == 0`.
    pub fn state(&self, frames_cap: usize) -> Result<MimiDecoderState> {
        if frames_cap == 0 {
            return Err(VokraError::InvalidArgument(
                "mimi decoder state: frames_cap must be > 0".into(),
            ));
        }
        let dim = self.config.seanet.dimension;
        let ds = self.frame_up.stride;
        let mut bufs = Vec::new();
        let mut tmp = Vec::new();
        let mut mid = Vec::new();
        // Edge 0: features at `dim` width [dim, frames].
        bufs.push(vec![0.0f32; dim * frames_cap]);
        // frame_up out [dim, frames*ds] (25 Hz).
        let mut t = frames_cap * ds;
        bufs.push(vec![0.0f32; dim * t]);
        let tf_rows = vec![0.0f32; t * dim];
        // init conv out.
        let mut ch = self.init_conv.out_ch;
        let init_state = self.init_conv.state(t);
        bufs.push(vec![0.0f32; ch * t]);
        let mut stage_states = Vec::new();
        for stage in &self.stages {
            let up_state = stage.up.state(t);
            t *= stage.up.stride;
            ch = stage.up.out_ch;
            bufs.push(vec![0.0f32; ch * t]);
            let hidden = stage.blocks.first().map_or(1, |b| b.conv1.out_ch);
            tmp.push(vec![0.0f32; ch * t]);
            mid.push(vec![0.0f32; hidden * t]);
            let mut block_states = Vec::new();
            for b in &stage.blocks {
                block_states.push((b.conv1.state(t), b.conv2.state(t)));
            }
            stage_states.push((up_state, block_states));
        }
        // final conv out [1, t].
        let final_state = self.final_conv.state(t);
        bufs.push(vec![0.0f32; t]);
        Ok(MimiDecoderState {
            frames_cap,
            frame_up: self.frame_up.state(frames_cap),
            tf: self.transformer.state(),
            init: init_state,
            stages: stage_states,
            final_conv: final_state,
            bufs,
            tmp,
            mid,
            tf_rows,
            proj: vec![0.0; dim],
        })
    }

    /// Decodes `features = [n_frames, feature_dim]` row-major into
    /// `pcm_out = [n_frames * frame_hop]`, carrying all causal state.
    /// Allocation-free.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a wrong feature width /
    /// capacity overflow / wrong `pcm_out` length.
    pub fn decode_into(
        &self,
        state: &mut MimiDecoderState,
        features: &[f32],
        pcm_out: &mut [f32],
    ) -> Result<()> {
        let feat = self.expected_feature_dim();
        if features.is_empty() || features.len() % feat != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "mimi decode: features len {} is not a positive multiple of the \
                 expected width {feat} (raw-table path = quantizer dim, \
                 effective-table path = seanet dimension — module docs)",
                features.len()
            )));
        }
        let n_frames = features.len() / feat;
        if n_frames > state.frames_cap {
            return Err(VokraError::InvalidArgument(format!(
                "mimi decode: {n_frames} frames > state capacity {}",
                state.frames_cap
            )));
        }
        let hop = self.frame_hop()?;
        if pcm_out.len() != n_frames * hop {
            return Err(VokraError::InvalidArgument(format!(
                "mimi decode: pcm_out len {} != n_frames*hop {}",
                pcm_out.len(),
                n_frames * hop
            )));
        }
        let compute = self.compute()?;
        let dim = self.config.seanet.dimension;

        // Stage features at `dim` width, channel-major [dim, n_frames].
        for f in 0..n_frames {
            let row = &features[f * feat..(f + 1) * feat];
            match &self.feature_proj {
                Some(w) => {
                    compute.gemv_f32(dim, feat, w, row, None, &mut state.proj)?;
                    for c in 0..dim {
                        state.bufs[0][c * n_frames + f] = state.proj[c];
                    }
                }
                None => {
                    for (c, v) in row.iter().enumerate() {
                        state.bufs[0][c * n_frames + f] = *v;
                    }
                }
            }
        }

        // Frame resample up (12.5 → 25 Hz).
        let ds = self.frame_up.stride;
        let mut t = n_frames * ds;
        {
            let (before, after) = state.bufs.split_at_mut(1);
            self.frame_up.process_into(
                &compute,
                &mut state.frame_up,
                &before[0][..dim * n_frames],
                n_frames,
                &mut after[0][..dim * t],
            )?;
        }
        let mut edge = 1;

        // Decoder transformer at 25 Hz.
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

        // SEANet decoder: init conv.
        let mut ch = self.init_conv.out_ch;
        {
            let (before, after) = state.bufs.split_at_mut(edge + 1);
            self.init_conv.process_into(
                &compute,
                &mut state.init,
                &before[edge][..dim * t],
                t,
                &mut after[0][..ch * t],
            )?;
        }
        edge += 1;

        for (si, stage) in self.stages.iter().enumerate() {
            let (up_state, block_states) = &mut state.stages[si];
            // ELU → transposed upsample into the next edge.
            elu_inplace(&mut state.bufs[edge][..ch * t]);
            let t_out = t * stage.up.stride;
            let out_ch = stage.up.out_ch;
            {
                let (before, after) = state.bufs.split_at_mut(edge + 1);
                stage.up.process_into(
                    &compute,
                    up_state,
                    &before[edge][..ch * t],
                    t,
                    &mut after[0][..out_ch * t_out],
                )?;
            }
            edge += 1;
            ch = out_ch;
            t = t_out;
            // Residual blocks in place.
            for (bi, block) in stage.blocks.iter().enumerate() {
                let hidden = block.conv1.out_ch;
                state.tmp[si][..ch * t].copy_from_slice(&state.bufs[edge][..ch * t]);
                elu_inplace(&mut state.tmp[si][..ch * t]);
                block.conv1.process_into(
                    &compute,
                    &mut block_states[bi].0,
                    &state.tmp[si][..ch * t],
                    t,
                    &mut state.mid[si][..hidden * t],
                )?;
                elu_inplace(&mut state.mid[si][..hidden * t]);
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
        }

        // ELU → final conv → [1, t] PCM.
        elu_inplace(&mut state.bufs[edge][..ch * t]);
        {
            let (before, after) = state.bufs.split_at_mut(edge + 1);
            self.final_conv.process_into(
                &compute,
                &mut state.final_conv,
                &before[edge][..ch * t],
                t,
                &mut after[0][..t],
            )?;
        }
        edge += 1;
        debug_assert_eq!(t, n_frames * hop);
        pcm_out.copy_from_slice(&state.bufs[edge][..t]);
        Ok(())
    }

    /// Convenience: fresh-state whole-buffer decode.
    ///
    /// # Errors
    ///
    /// See [`Self::decode_into`].
    pub fn decode_all(&self, features: &[f32]) -> Result<Vec<f32>> {
        let feat = self.expected_feature_dim();
        if features.is_empty() || features.len() % feat != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "mimi decode: features len {} is not a positive multiple of {feat}",
                features.len()
            )));
        }
        let n_frames = features.len() / feat;
        let mut state = self.state(n_frames)?;
        let mut pcm = vec![0.0f32; n_frames * self.frame_hop()?];
        self.decode_into(&mut state, features, &mut pcm)?;
        Ok(pcm)
    }
}

/// Reads a `[in, out, k]` transposed causal conv (+ optional `{name}.bias`)
/// from the GGUF and builds it (Vokra structural naming — see
/// [`MimiNeuralDecoder::from_gguf`] for the honest naming boundary).
fn read_convtr(
    file: &GgufFile,
    name: &str,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    stride: usize,
    bias: bool,
) -> Result<CausalConvTranspose1d> {
    let w = tensor_f32(file, &format!("{name}.weight"), in_ch * out_ch * k)?;
    let b = if bias {
        Some(tensor_f32(file, &format!("{name}.bias"), out_ch)?)
    } else {
        None
    };
    CausalConvTranspose1d::new(in_ch, out_ch, k, stride, w, b)
}

/// Writes a transposed conv's `[in, out, k]` weight (+ bias) under
/// `{name}.*` (round-trip mirror of [`read_convtr`]).
#[cfg(test)]
fn write_convtr(b: &mut vokra_core::gguf::GgufBuilder, name: &str, c: &CausalConvTranspose1d) {
    use vokra_core::gguf::GgmlType;
    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let w = c.weight_iok();
    b.add_tensor(
        &format!("{name}.weight"),
        GgmlType::F32,
        vec![w.len() as u64],
        bytes(w),
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

/// Packs **every** decoder tensor under the structural `mimi.dec.*` naming
/// (the exact round-trip mirror of [`MimiNeuralDecoder::from_gguf`]) — test
/// support for the engine-level side-car bind tests (`moshi::engine`); the
/// encoder-side companion is `encoder::pack_encoder_structural`.
#[cfg(test)]
pub(crate) fn pack_decoder_structural(
    b: &mut vokra_core::gguf::GgufBuilder,
    src: &MimiNeuralDecoder,
) {
    use super::encoder::{write_conv, write_tf_layer};
    use vokra_core::gguf::GgmlType;
    if let Some(fp) = &src.feature_proj {
        let bytes: Vec<u8> = fp.iter().flat_map(|x| x.to_le_bytes()).collect();
        b.add_tensor(
            "mimi.dec.feature_proj",
            GgmlType::F32,
            vec![fp.len() as u64],
            bytes,
        )
        .unwrap();
    }
    write_convtr(b, "mimi.dec.frame_up", &src.frame_up);
    for (l, layer) in src.transformer.layers.iter().enumerate() {
        write_tf_layer(b, &format!("mimi.dec.tf{l}"), layer);
    }
    write_conv(b, "mimi.dec.init", &src.init_conv);
    for (i, stage) in src.stages.iter().enumerate() {
        write_convtr(b, &format!("mimi.dec.s{i}.up"), &stage.up);
        for (j, blk) in stage.blocks.iter().enumerate() {
            write_conv(b, &format!("mimi.dec.s{i}.b{j}.c1"), &blk.conv1);
            write_conv(b, &format!("mimi.dec.s{i}.b{j}.c2"), &blk.conv2);
        }
    }
    write_conv(b, "mimi.dec.final", &src.final_conv);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoder(project: bool) -> MimiNeuralDecoder {
        MimiNeuralDecoder::synthesized(&MimiNeuralConfig::tiny_for_tests(), 9, project)
            .expect("decoder")
    }

    fn features(n_frames: usize, width: usize) -> Vec<f32> {
        (0..n_frames * width)
            .map(|i| (i as f32 * 0.31).sin() * 0.4)
            .collect()
    }

    #[test]
    fn decode_shapes_determinism_and_bounded_output() {
        let dec = decoder(true);
        let hop = dec.frame_hop().unwrap();
        let x = features(3, dec.expected_feature_dim());
        let a = dec.decode_all(&x).unwrap();
        let b = dec.decode_all(&x).unwrap();
        assert_eq!(a.len(), 3 * hop);
        assert_eq!(a, b, "deterministic");
        assert!(a.iter().all(|v| v.is_finite()));
        // The chain must respond to its input (energy flows end to end).
        let y = features(3, dec.expected_feature_dim())
            .iter()
            .map(|v| v * 2.0)
            .collect::<Vec<_>>();
        let c = dec.decode_all(&y).unwrap();
        assert_ne!(a, c, "different features must produce different PCM");
    }

    #[test]
    fn full_buffer_equals_frame_streaming() {
        // T33: full-buffer decode == per-frame decode with carried state.
        let dec = decoder(true);
        let hop = dec.frame_hop().unwrap();
        let feat = dec.expected_feature_dim();
        let x = features(4, feat);
        let full = dec.decode_all(&x).unwrap();
        let mut st = dec.state(1).unwrap();
        let mut streamed = Vec::new();
        for f in 0..4 {
            let mut pcm = vec![0.0f32; hop];
            dec.decode_into(&mut st, &x[f * feat..(f + 1) * feat], &mut pcm)
                .unwrap();
            streamed.extend_from_slice(&pcm);
        }
        assert_eq!(full.len(), streamed.len());
        for (i, (a, b)) in full.iter().zip(streamed.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "sample {i}: full {a} vs streamed {b} — boundary state carry-over broke"
            );
        }
    }

    #[test]
    fn effective_table_path_skips_the_projection() {
        let dec = decoder(false);
        assert_eq!(
            dec.expected_feature_dim(),
            dec.config().seanet.dimension,
            "effective-table path takes seanet-width features"
        );
        let x = features(2, dec.expected_feature_dim());
        let pcm = dec.decode_all(&x).unwrap();
        assert_eq!(pcm.len(), 2 * dec.frame_hop().unwrap());
    }

    #[test]
    fn wrong_feature_width_is_loud() {
        let dec = decoder(true);
        let feat = dec.expected_feature_dim();
        assert!(dec.decode_all(&features(1, feat)[..feat - 1]).is_err());
        assert!(dec.decode_all(&[]).is_err());
        let mut st = dec.state(1).unwrap();
        let mut pcm = vec![0.0f32; dec.frame_hop().unwrap()];
        // Two frames into a 1-frame state.
        assert!(
            dec.decode_into(&mut st, &features(2, feat), &mut pcm)
                .is_err()
        );
    }

    #[test]
    fn from_gguf_binds_structural_named_tensors_round_trip() {
        // Pack every neural-chain weight under the Vokra structural naming
        // (`mimi.dec.*`), then verify `from_gguf` binds it back to a decoder
        // that reproduces the exact decode. Pack ↔ unpack self-consistency;
        // real kyutai-name / weight_norm mapping stays owner (module docs).
        use super::super::encoder::{write_conv, write_tf_layer};
        use vokra_core::gguf::{GgmlType, GgufBuilder};
        let cfg = MimiNeuralConfig::tiny_for_tests();
        let src = MimiNeuralDecoder::synthesized(&cfg, 9, true).unwrap();

        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "mimi");
        let write_vec = |b: &mut GgufBuilder, name: &str, v: &[f32]| {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            b.add_tensor(name, GgmlType::F32, vec![v.len() as u64], bytes)
                .unwrap();
        };
        if let Some(fp) = &src.feature_proj {
            write_vec(&mut b, "mimi.dec.feature_proj", fp);
        }
        write_convtr(&mut b, "mimi.dec.frame_up", &src.frame_up);
        for (l, layer) in src.transformer.layers.iter().enumerate() {
            write_tf_layer(&mut b, &format!("mimi.dec.tf{l}"), layer);
        }
        write_conv(&mut b, "mimi.dec.init", &src.init_conv);
        for (i, stage) in src.stages.iter().enumerate() {
            write_convtr(&mut b, &format!("mimi.dec.s{i}.up"), &stage.up);
            for (j, blk) in stage.blocks.iter().enumerate() {
                write_conv(&mut b, &format!("mimi.dec.s{i}.b{j}.c1"), &blk.conv1);
                write_conv(&mut b, &format!("mimi.dec.s{i}.b{j}.c2"), &blk.conv2);
            }
        }
        write_conv(&mut b, "mimi.dec.final", &src.final_conv);

        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let loaded = MimiNeuralDecoder::from_gguf(&file, &cfg).expect("bind");
        assert!(!loaded.is_synthesized());
        assert_eq!(loaded.expected_feature_dim(), src.expected_feature_dim());

        // Forward equality: identical features → identical PCM.
        let x = features(3, src.expected_feature_dim());
        let a = src.decode_all(&x).unwrap();
        let e = loaded.decode_all(&x).unwrap();
        assert_eq!(a, e, "structural pack → GGUF → bind reproduces the decode");
    }

    #[test]
    fn from_gguf_missing_tensor_is_a_loud_model_load_error() {
        // FR-EX-08: feature_proj is optional (absent → None), so the first
        // required missing tensor is frame_up — a loud ModelLoad naming it.
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "mimi");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let err =
            MimiNeuralDecoder::from_gguf(&file, &MimiNeuralConfig::tiny_for_tests()).unwrap_err();
        assert!(matches!(err, VokraError::ModelLoad(_)));
        assert!(
            err.to_string().contains("mimi.dec.frame_up"),
            "names the missing tensor: {err}"
        );
    }
}
