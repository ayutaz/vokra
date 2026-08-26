//! Shared AudioCraft autoregressive LM for the public MusicGen-Medium/Large
//! and AudioGen-Medium GGUF layouts.
//!
//! The public artifacts keep AudioCraft's upstream tensor names verbatim. This
//! module binds that exact LM-only layout, leaves embeddings, output heads and
//! transformer blocks in the GGUF mapping, and widens one layer at a time into
//! a reused scratch block. It deliberately does not tokenize text or decode
//! EnCodec waveform samples: callers supply T5 hidden states and receive four
//! codebook-logit rows. Those two companion boundaries remain explicit.
//!
//! All learned reductions use [`Compute`]. CPU is the reference backend;
//! Metal uses the existing fused attention and MLP paths. Embedding lookup,
//! sinusoidal position construction, residual addition and delay-pattern
//! scheduling are deterministic host-side layout glue, not a hidden CPU model
//! fallback.

use std::sync::{Arc, Mutex, MutexGuard};

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile, GgufTensorInfo};
use vokra_core::{KvCache, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::mapped_weights::{MappedModel, lock_scratch, mapped_info, transpose_widen, widen_into};
use crate::whisper::nn::{
    add_assign, attention_from_kv_into, layer_norm_into, mlp_into, project_kv_into,
};
use crate::whisper::scratch::BlockScratch;
use crate::whisper::weights::{Attention, DecoderLayer, LayerNorm, Linear, LinearWeight};

const AUDIOCRAFT_MAPPED: MappedModel = MappedModel {
    name: "audiocraft-lm",
    resident_entry: "AudioCraftLmDecoder::bind (dense public AudioCraft GGUF required)",
};

/// AudioCraft `StreamingTransformer` sinusoidal maximum period.
const POSITION_MAX_PERIOD: f32 = 10_000.0;
/// Rows widened at once for one mapped codebook head GEMV.
const HEAD_CHUNK_ROWS: usize = 128;

/// Every learned operation required by the AudioCraft autoregressive LM.
///
/// [`Compute::for_backend`] checks this set before condition preparation or a
/// decode step starts. An unsupported backend therefore fails before partial
/// state mutation and never falls back to CPU.
pub const AUDIOCRAFT_LM_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
];

/// Shape contract for one AudioCraft LM-only checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioCraftLmConfig {
    pub d_model: usize,
    pub num_layers: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub num_codebooks: usize,
}

impl AudioCraftLmConfig {
    /// Validates the axes used by the native forward.
    pub fn validate(self) -> Result<()> {
        if self.d_model < 4
            || self.d_model % 2 != 0
            || self.num_layers == 0
            || self.n_heads == 0
            || self.d_model % self.n_heads != 0
            || self.ffn_dim == 0
            || self.vocab_size == 0
            || self.vocab_size > u32::MAX as usize
            || self.num_codebooks == 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft LM config is invalid: d_model={}, num_layers={}, n_heads={}, \
                 ffn_dim={}, vocab_size={}, num_codebooks={} (d_model must be even and \
                 divisible by n_heads; every axis must be non-zero; vocab_size must fit u32)",
                self.d_model,
                self.num_layers,
                self.n_heads,
                self.ffn_dim,
                self.vocab_size,
                self.num_codebooks
            )));
        }
        Ok(())
    }

    #[must_use]
    pub const fn special_token_id(self) -> u32 {
        self.vocab_size as u32
    }
}

/// T5 condition after AudioCraft's learned `description.output_proj` and mask.
#[derive(Debug, Clone)]
pub struct AudioCraftCondition {
    projected: Vec<f32>,
    frames: usize,
    d_model: usize,
    decoder_identity: Arc<()>,
}

impl AudioCraftCondition {
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.d_model
    }

    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.projected
    }
}

struct MappedLayerLocs {
    self_in: GgufTensorInfo,
    self_out: GgufTensorInfo,
    cross_in: GgufTensorInfo,
    cross_out: GgufTensorInfo,
    norm1_w: GgufTensorInfo,
    norm1_b: GgufTensorInfo,
    norm_cross_w: GgufTensorInfo,
    norm_cross_b: GgufTensorInfo,
    norm2_w: GgufTensorInfo,
    norm2_b: GgufTensorInfo,
    fc1: GgufTensorInfo,
    fc2: GgufTensorInfo,
}

struct MappedHeads {
    embeddings: Vec<GgufTensorInfo>,
    linears: Vec<GgufTensorInfo>,
    chunk: Mutex<Vec<f32>>,
}

struct MappedWeights {
    file: Arc<GgufFile>,
    layers: Vec<MappedLayerLocs>,
    layer_scratch: Mutex<DecoderLayer>,
    heads: MappedHeads,
    condition_proj: Linear,
    out_norm: LayerNorm,
    condition_dim: usize,
    config: AudioCraftLmConfig,
}

/// Bounded-memory AudioCraft autoregressive decoder.
pub struct AudioCraftLmDecoder {
    weights: MappedWeights,
    backend: BackendKind,
    identity: Arc<()>,
}

impl std::fmt::Debug for AudioCraftLmDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioCraftLmDecoder")
            .field("config", &self.weights.config)
            .field("condition_dim", &self.weights.condition_dim)
            .field("backend", &self.backend)
            .finish()
    }
}

impl AudioCraftLmDecoder {
    /// Binds the exact AudioCraft LM-only tensor layout from a mapped GGUF.
    ///
    /// The caller must first authenticate the model-family manifest. This
    /// binder independently validates every tensor name, dense dtype and exact
    /// shape needed by execution so malformed newer conversions still fail at
    /// load rather than halfway through generation.
    pub fn bind(
        file: Arc<GgufFile>,
        config: AudioCraftLmConfig,
        backend: BackendKind,
    ) -> Result<Self> {
        config.validate()?;
        Compute::for_backend(backend, AUDIOCRAFT_LM_HOT_OPS)?;
        let d = config.d_model;
        let ffn = config.ffn_dim;
        let rows = config.vocab_size + 1;

        let condition_weight_name =
            "condition_provider.conditioners.description.output_proj.weight";
        let raw_condition = file.tensor_info(condition_weight_name).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "audiocraft-lm: tensor `{condition_weight_name}`: missing"
            ))
        })?;
        let condition_dims = dims(raw_condition);
        if condition_dims.len() != 2 || condition_dims[0] != d || condition_dims[1] == 0 {
            return Err(VokraError::ModelLoad(format!(
                "audiocraft-lm: tensor `{condition_weight_name}` has dimensions \
                 {condition_dims:?}, expected [{d}, condition_dim]"
            )));
        }
        let condition_dim = condition_dims[1];
        let condition_info = exact_info(&file, condition_weight_name, &[d, condition_dim])?;
        let condition_bias = exact_info(
            &file,
            "condition_provider.conditioners.description.output_proj.bias",
            &[d],
        )?;
        let mut condition_w_t = Vec::new();
        transpose_widen(
            file.tensor_bytes(&condition_info),
            condition_info.dtype,
            d,
            condition_dim,
            &mut condition_w_t,
            AUDIOCRAFT_MAPPED,
        )?;
        let mut condition_b = Vec::new();
        widen_into(
            file.tensor_bytes(&condition_bias),
            condition_bias.dtype,
            &mut condition_b,
            AUDIOCRAFT_MAPPED,
        )?;

        let out_norm_w = exact_info(&file, "out_norm.weight", &[d])?;
        let out_norm_b = exact_info(&file, "out_norm.bias", &[d])?;
        let mut out_gamma = Vec::new();
        let mut out_beta = Vec::new();
        widen_into(
            file.tensor_bytes(&out_norm_w),
            out_norm_w.dtype,
            &mut out_gamma,
            AUDIOCRAFT_MAPPED,
        )?;
        widen_into(
            file.tensor_bytes(&out_norm_b),
            out_norm_b.dtype,
            &mut out_beta,
            AUDIOCRAFT_MAPPED,
        )?;

        let mut embeddings = Vec::with_capacity(config.num_codebooks);
        let mut linears = Vec::with_capacity(config.num_codebooks);
        for codebook in 0..config.num_codebooks {
            embeddings.push(exact_info(
                &file,
                &format!("emb.{codebook}.weight"),
                &[rows, d],
            )?);
            linears.push(exact_info(
                &file,
                &format!("linears.{codebook}.weight"),
                &[config.vocab_size, d],
            )?);
        }

        let mut layers = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let p = format!("transformer.layers.{layer}");
            layers.push(MappedLayerLocs {
                self_in: exact_info(&file, &format!("{p}.self_attn.in_proj_weight"), &[3 * d, d])?,
                self_out: exact_info(&file, &format!("{p}.self_attn.out_proj.weight"), &[d, d])?,
                cross_in: exact_info(
                    &file,
                    &format!("{p}.cross_attention.in_proj_weight"),
                    &[3 * d, d],
                )?,
                cross_out: exact_info(
                    &file,
                    &format!("{p}.cross_attention.out_proj.weight"),
                    &[d, d],
                )?,
                norm1_w: exact_info(&file, &format!("{p}.norm1.weight"), &[d])?,
                norm1_b: exact_info(&file, &format!("{p}.norm1.bias"), &[d])?,
                norm_cross_w: exact_info(&file, &format!("{p}.norm_cross.weight"), &[d])?,
                norm_cross_b: exact_info(&file, &format!("{p}.norm_cross.bias"), &[d])?,
                norm2_w: exact_info(&file, &format!("{p}.norm2.weight"), &[d])?,
                norm2_b: exact_info(&file, &format!("{p}.norm2.bias"), &[d])?,
                fc1: exact_info(&file, &format!("{p}.linear1.weight"), &[ffn, d])?,
                fc2: exact_info(&file, &format!("{p}.linear2.weight"), &[d, ffn])?,
            });
        }

        Ok(Self {
            weights: MappedWeights {
                file,
                layers,
                layer_scratch: Mutex::new(empty_layer(d, ffn)),
                heads: MappedHeads {
                    embeddings,
                    linears,
                    chunk: Mutex::new(Vec::new()),
                },
                condition_proj: Linear::dense(condition_w_t, condition_dim, d, Some(condition_b)),
                out_norm: LayerNorm {
                    gamma: out_gamma,
                    beta: out_beta,
                },
                condition_dim,
                config,
            },
            backend,
            identity: Arc::new(()),
        })
    }

    #[must_use]
    pub const fn config(&self) -> AudioCraftLmConfig {
        self.weights.config
    }

    #[must_use]
    pub const fn condition_dim(&self) -> usize {
        self.weights.condition_dim
    }

    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Applies the checkpoint's learned T5 output projection and upstream mask.
    ///
    /// `hidden` is `[frames, condition_dim]`. `mask`, when provided, contains
    /// exactly `frames` values in `{0,1}`. AudioCraft applies the linear first
    /// and then zeros masked rows; preserving that order matters because the
    /// projection has a bias.
    pub fn prepare_condition(
        &self,
        hidden: &[f32],
        frames: usize,
        mask: Option<&[u8]>,
    ) -> Result<AudioCraftCondition> {
        let condition_dim = self.weights.condition_dim;
        if frames == 0 || hidden.len() != frames.saturating_mul(condition_dim) {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft condition: hidden len {} != frames {} * condition_dim {}",
                hidden.len(),
                frames,
                condition_dim
            )));
        }
        if hidden.iter().any(|v| !v.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "audiocraft condition: hidden states contain a non-finite value".to_owned(),
            ));
        }
        if let Some(mask) = mask {
            if mask.len() != frames || mask.iter().any(|&value| value > 1) {
                return Err(VokraError::InvalidArgument(format!(
                    "audiocraft condition: mask must contain {frames} binary entries"
                )));
            }
        }

        let d = self.weights.config.d_model;
        let mut projected = vec![0.0; frames * d];
        let compute = self.compute()?;
        crate::whisper::nn::linear_apply(
            &compute,
            &mut projected,
            hidden,
            frames,
            &self.weights.condition_proj,
        )?;
        if let Some(mask) = mask {
            for (frame, &visible) in mask.iter().enumerate() {
                if visible == 0 {
                    projected[frame * d..(frame + 1) * d].fill(0.0);
                }
            }
        }
        Ok(AudioCraftCondition {
            projected,
            frames,
            d_model: d,
            decoder_identity: Arc::clone(&self.identity),
        })
    }

    /// Creates an autoregressive state and precomputes each layer's fixed
    /// cross-attention K/V from `condition`.
    pub fn new_state(
        &self,
        condition: &AudioCraftCondition,
        max_steps: usize,
    ) -> Result<AudioCraftLmState> {
        let cfg = self.weights.config;
        if max_steps == 0 {
            return Err(VokraError::InvalidArgument(
                "audiocraft LM state: max_steps must be non-zero".to_owned(),
            ));
        }
        if condition.frames == 0
            || condition.d_model != cfg.d_model
            || condition.projected.len() != condition.frames * cfg.d_model
        {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft LM state: condition shape [{}, {}] does not match d_model {}",
                condition.frames, condition.d_model, cfg.d_model
            )));
        }
        if !Arc::ptr_eq(&condition.decoder_identity, &self.identity) {
            return Err(VokraError::InvalidArgument(
                "audiocraft LM state: condition belongs to a different decoder/checkpoint"
                    .to_owned(),
            ));
        }

        let compute = self.compute()?;
        let mut cross_kv = Vec::with_capacity(cfg.num_layers);
        let mut guard = self.lock_layer_scratch()?;
        for layer_index in 0..cfg.num_layers {
            let layer = self.materialize_layer(&mut guard, layer_index)?;
            let mut k = Vec::new();
            let mut v = Vec::new();
            project_kv_into(
                &compute,
                &mut k,
                &mut v,
                &condition.projected,
                condition.frames,
                &layer.cross_attn,
            )?;
            cross_kv.push((k, v));
        }
        drop(guard);

        Ok(AudioCraftLmState {
            self_kv: KvCache::with_reserve(cfg.num_layers, cfg.d_model, max_steps),
            cross_kv,
            condition_frames: condition.frames,
            max_steps,
            config: cfg,
            decoder_identity: Arc::clone(&self.identity),
            h: vec![0.0; cfg.d_model],
            block: BlockScratch::with_reserve(
                1,
                max_steps.max(condition.frames),
                cfg.d_model,
                cfg.ffn_dim,
                cfg.n_heads,
            ),
            poisoned: false,
        })
    }

    /// Advances one delayed-sequence position and writes codebook-major logits
    /// `[num_codebooks, vocab_size]` into `logits`.
    pub fn step_into(
        &self,
        state: &mut AudioCraftLmState,
        tokens: &[u32],
        logits: &mut [f32],
    ) -> Result<()> {
        let cfg = self.weights.config;
        state.validate_for(cfg, &self.identity)?;
        if tokens.len() != cfg.num_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft LM step: {} tokens != num_codebooks {}",
                tokens.len(),
                cfg.num_codebooks
            )));
        }
        let expected_logits = cfg.num_codebooks * cfg.vocab_size;
        if logits.len() != expected_logits {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft LM step: logits len {} != num_codebooks * vocab_size {}",
                logits.len(),
                expected_logits
            )));
        }
        let start = state.self_kv.positions();
        if start >= state.max_steps {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft LM step: position {start} reached max_steps {} (no silent wrap)",
                state.max_steps
            )));
        }
        for (codebook, &token) in tokens.iter().enumerate() {
            if token as usize > cfg.vocab_size {
                return Err(VokraError::InvalidArgument(format!(
                    "audiocraft LM step: codebook {codebook} token {token} exceeds special token {}",
                    cfg.special_token_id()
                )));
            }
        }

        state.h.fill(0.0);
        self.embed_tokens_into(tokens, &mut state.h)?;
        add_sinusoidal_position(&mut state.h, start)?;

        let compute = self.compute()?;
        let mut guard = self.lock_layer_scratch()?;
        let t_kv = start + 1;
        // Any failure from here can leave one or more layer K/V rows appended.
        // Keep that partial state unusable until an explicit reset instead of
        // allowing a retry to duplicate rows behind the unchanged position.
        state.poisoned = true;
        for layer_index in 0..cfg.num_layers {
            let layer = self.materialize_layer(&mut guard, layer_index)?;
            state.block.ensure_residual(1, cfg.d_model, cfg.ffn_dim);

            layer_norm_into(&compute, &mut state.block.ln, &state.h, 1, &layer.self_ln)?;
            project_kv_into(
                &compute,
                &mut state.block.k,
                &mut state.block.v,
                &state.block.ln,
                1,
                &layer.self_attn,
            )?;
            state
                .self_kv
                .append(layer_index, &state.block.k, &state.block.v);
            // A one-position step has no future row in its cache. Marking the
            // attention non-causal is mathematically identical and permits the
            // Metal fused-attention route; the cache boundary enforces causality.
            attention_from_kv_into(
                &compute,
                &mut state.block.attn,
                &state.block.ln,
                1,
                state.self_kv.k(layer_index),
                state.self_kv.v(layer_index),
                t_kv,
                &layer.self_attn.q,
                &layer.self_attn.out,
                cfg.n_heads,
                false,
                0,
                &mut state.block.block_out,
            )?;
            add_assign(&mut state.h, &state.block.block_out)?;

            layer_norm_into(&compute, &mut state.block.ln, &state.h, 1, &layer.cross_ln)?;
            let (cross_k, cross_v) = &state.cross_kv[layer_index];
            attention_from_kv_into(
                &compute,
                &mut state.block.attn,
                &state.block.ln,
                1,
                cross_k,
                cross_v,
                state.condition_frames,
                &layer.cross_attn.q,
                &layer.cross_attn.out,
                cfg.n_heads,
                false,
                0,
                &mut state.block.block_out,
            )?;
            add_assign(&mut state.h, &state.block.block_out)?;

            layer_norm_into(&compute, &mut state.block.ln, &state.h, 1, &layer.mlp_ln)?;
            mlp_into(
                &compute,
                &mut state.block.mlp_h,
                &mut state.block.mlp_a,
                &mut state.block.block_out,
                &state.block.ln,
                1,
                &layer.fc1,
                &layer.fc2,
            )?;
            add_assign(&mut state.h, &state.block.block_out)?;
        }
        layer_norm_into(
            &compute,
            &mut state.block.ln,
            &state.h,
            1,
            &self.weights.out_norm,
        )?;
        self.logits_into(&compute, &state.block.ln, logits)?;
        state.self_kv.advance(1);
        state.poisoned = false;
        Ok(())
    }

    fn compute(&self) -> Result<Compute> {
        Compute::for_backend(self.backend, AUDIOCRAFT_LM_HOT_OPS)
    }

    fn lock_layer_scratch(&self) -> Result<MutexGuard<'_, DecoderLayer>> {
        lock_scratch(&self.weights.layer_scratch, AUDIOCRAFT_MAPPED)
    }

    fn materialize_layer<'a>(
        &self,
        scratch: &'a mut DecoderLayer,
        layer_index: usize,
    ) -> Result<&'a DecoderLayer> {
        let cfg = self.weights.config;
        let d = cfg.d_model;
        let ffn = cfg.ffn_dim;
        let locs = self.weights.layers.get(layer_index).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "audiocraft LM: layer {layer_index} >= {}",
                cfg.num_layers
            ))
        })?;
        let f = &self.weights.file;

        materialize_norm(f, &locs.norm1_w, &locs.norm1_b, &mut scratch.self_ln)?;
        materialize_attention(f, &locs.self_in, &locs.self_out, d, &mut scratch.self_attn)?;
        materialize_norm(
            f,
            &locs.norm_cross_w,
            &locs.norm_cross_b,
            &mut scratch.cross_ln,
        )?;
        materialize_attention(
            f,
            &locs.cross_in,
            &locs.cross_out,
            d,
            &mut scratch.cross_attn,
        )?;
        materialize_norm(f, &locs.norm2_w, &locs.norm2_b, &mut scratch.mlp_ln)?;
        transpose_widen(
            f.tensor_bytes(&locs.fc1),
            locs.fc1.dtype,
            ffn,
            d,
            dense_weight_mut(&mut scratch.fc1)?,
            AUDIOCRAFT_MAPPED,
        )?;
        transpose_widen(
            f.tensor_bytes(&locs.fc2),
            locs.fc2.dtype,
            d,
            ffn,
            dense_weight_mut(&mut scratch.fc2)?,
            AUDIOCRAFT_MAPPED,
        )?;
        scratch.fc1.bias = None;
        scratch.fc2.bias = None;
        Ok(scratch)
    }

    fn embed_tokens_into(&self, tokens: &[u32], out: &mut [f32]) -> Result<()> {
        let d = self.weights.config.d_model;
        for (codebook, &token) in tokens.iter().enumerate() {
            add_mapped_row(
                &self.weights.file,
                &self.weights.heads.embeddings[codebook],
                token as usize,
                d,
                out,
            )?;
        }
        Ok(())
    }

    fn logits_into(&self, compute: &Compute, hidden: &[f32], out: &mut [f32]) -> Result<()> {
        let cfg = self.weights.config;
        let d = cfg.d_model;
        let vocab = cfg.vocab_size;
        let mut chunk = lock_scratch(&self.weights.heads.chunk, AUDIOCRAFT_MAPPED)?;
        for (codebook, info) in self.weights.heads.linears.iter().enumerate() {
            let bytes = self.weights.file.tensor_bytes(info);
            let esz = info.dtype.type_size();
            let codebook_out = &mut out[codebook * vocab..(codebook + 1) * vocab];
            let mut row = 0usize;
            while row < vocab {
                let rows = HEAD_CHUNK_ROWS.min(vocab - row);
                let n = rows * d;
                let start = row * d * esz;
                let end = start + n * esz;
                widen_into(
                    &bytes[start..end],
                    info.dtype,
                    &mut chunk,
                    AUDIOCRAFT_MAPPED,
                )?;
                compute.gemv_f32(
                    rows,
                    d,
                    &chunk,
                    hidden,
                    None,
                    &mut codebook_out[row..row + rows],
                )?;
                row += rows;
            }
        }
        Ok(())
    }
}

/// Mutable state for one AudioCraft autoregressive stream.
pub struct AudioCraftLmState {
    self_kv: KvCache,
    cross_kv: Vec<(Vec<f32>, Vec<f32>)>,
    condition_frames: usize,
    max_steps: usize,
    config: AudioCraftLmConfig,
    decoder_identity: Arc<()>,
    h: Vec<f32>,
    block: BlockScratch,
    poisoned: bool,
}

impl std::fmt::Debug for AudioCraftLmState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioCraftLmState")
            .field("position", &self.self_kv.positions())
            .field("condition_frames", &self.condition_frames)
            .field("max_steps", &self.max_steps)
            .field("config", &self.config)
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

impl AudioCraftLmState {
    #[must_use]
    pub fn position(&self) -> usize {
        self.self_kv.positions()
    }

    #[must_use]
    pub const fn max_steps(&self) -> usize {
        self.max_steps
    }

    /// Rewinds only the autoregressive self-attention cache. Fixed condition
    /// K/V remains valid and allocated.
    pub fn reset(&mut self) {
        self.self_kv.reset();
        self.h.fill(0.0);
        self.poisoned = false;
    }

    fn validate_for(&self, config: AudioCraftLmConfig, decoder_identity: &Arc<()>) -> Result<()> {
        if self.poisoned {
            return Err(VokraError::InvalidArgument(
                "audiocraft LM state is poisoned by a failed partial step; call reset() before retrying"
                    .to_owned(),
            ));
        }
        if self.config != config || self.cross_kv.len() != config.num_layers {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft LM state belongs to {:?}, decoder is {:?}",
                self.config, config
            )));
        }
        if !Arc::ptr_eq(&self.decoder_identity, decoder_identity) {
            return Err(VokraError::InvalidArgument(
                "audiocraft LM state belongs to a different decoder/checkpoint".to_owned(),
            ));
        }
        Ok(())
    }
}

fn dims(info: &GgufTensorInfo) -> Vec<usize> {
    info.dimensions.iter().map(|&axis| axis as usize).collect()
}

fn exact_info(file: &GgufFile, name: &str, expected: &[usize]) -> Result<GgufTensorInfo> {
    let count = expected.iter().try_fold(1usize, |acc, &axis| {
        acc.checked_mul(axis).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "audiocraft-lm: expected shape for `{name}` overflows usize: {expected:?}"
            ))
        })
    })?;
    let info = mapped_info(file, name, count, AUDIOCRAFT_MAPPED)?;
    let actual = dims(&info);
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "audiocraft-lm: tensor `{name}` has dimensions {actual:?}, expected {expected:?}"
        )));
    }
    Ok(info)
}

fn empty_linear(input: usize, output: usize) -> Linear {
    Linear::dense(Vec::new(), input, output, None)
}

fn empty_attention(d: usize) -> Attention {
    Attention {
        q: empty_linear(d, d),
        k: empty_linear(d, d),
        v: empty_linear(d, d),
        out: empty_linear(d, d),
    }
}

fn empty_norm() -> LayerNorm {
    LayerNorm {
        gamma: Vec::new(),
        beta: Vec::new(),
    }
}

fn empty_layer(d: usize, ffn: usize) -> DecoderLayer {
    DecoderLayer {
        self_ln: empty_norm(),
        self_attn: empty_attention(d),
        cross_ln: empty_norm(),
        cross_attn: empty_attention(d),
        mlp_ln: empty_norm(),
        fc1: empty_linear(d, ffn),
        fc2: empty_linear(ffn, d),
    }
}

fn dense_weight_mut(linear: &mut Linear) -> Result<&mut Vec<f32>> {
    match &mut linear.w {
        LinearWeight::Dense(weight) => Ok(weight),
        LinearWeight::KQuant { .. } => Err(VokraError::InvalidArgument(
            "audiocraft LM mapped scratch unexpectedly contains a quantized Linear".to_owned(),
        )),
    }
}

fn materialize_norm(
    file: &GgufFile,
    weight: &GgufTensorInfo,
    bias: &GgufTensorInfo,
    out: &mut LayerNorm,
) -> Result<()> {
    widen_into(
        file.tensor_bytes(weight),
        weight.dtype,
        &mut out.gamma,
        AUDIOCRAFT_MAPPED,
    )?;
    widen_into(
        file.tensor_bytes(bias),
        bias.dtype,
        &mut out.beta,
        AUDIOCRAFT_MAPPED,
    )
}

fn materialize_attention(
    file: &GgufFile,
    in_proj: &GgufTensorInfo,
    out_proj: &GgufTensorInfo,
    d: usize,
    out: &mut Attention,
) -> Result<()> {
    let bytes = file.tensor_bytes(in_proj);
    let esz = in_proj.dtype.type_size();
    let matrix_bytes = d * d * esz;
    for (index, linear) in [&mut out.q, &mut out.k, &mut out.v].into_iter().enumerate() {
        let start = index * matrix_bytes;
        transpose_widen(
            &bytes[start..start + matrix_bytes],
            in_proj.dtype,
            d,
            d,
            dense_weight_mut(linear)?,
            AUDIOCRAFT_MAPPED,
        )?;
        linear.bias = None;
    }
    transpose_widen(
        file.tensor_bytes(out_proj),
        out_proj.dtype,
        d,
        d,
        dense_weight_mut(&mut out.out)?,
        AUDIOCRAFT_MAPPED,
    )?;
    out.out.bias = None;
    Ok(())
}

fn add_mapped_row(
    file: &GgufFile,
    info: &GgufTensorInfo,
    row: usize,
    d: usize,
    out: &mut [f32],
) -> Result<()> {
    if out.len() != d {
        return Err(VokraError::InvalidArgument(format!(
            "audiocraft mapped embedding: out len {} != d_model {d}",
            out.len()
        )));
    }
    let rows = info.element_count().map_err(|error| {
        VokraError::ModelLoad(format!(
            "audiocraft mapped embedding `{}`: {error}",
            info.name
        ))
    })? as usize
        / d;
    if row >= rows {
        return Err(VokraError::InvalidArgument(format!(
            "audiocraft mapped embedding `{}`: row {row} >= rows {rows}",
            info.name
        )));
    }
    let esz = info.dtype.type_size();
    let bytes = file.tensor_bytes(info);
    let start = row * d * esz;
    let src = &bytes[start..start + d * esz];
    match info.dtype {
        GgmlType::F32 => {
            for (dst, bytes) in out.iter_mut().zip(src.chunks_exact(4)) {
                *dst += f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            }
        }
        GgmlType::F16 => {
            for (dst, bytes) in out.iter_mut().zip(src.chunks_exact(2)) {
                *dst +=
                    vokra_core::gguf::quant::f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
            }
        }
        GgmlType::BF16 => {
            for (dst, bytes) in out.iter_mut().zip(src.chunks_exact(2)) {
                *dst += f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16);
            }
        }
        other => {
            return Err(VokraError::ModelLoad(format!(
                "audiocraft mapped embedding `{}`: unsupported dtype {other:?}",
                info.name
            )));
        }
    }
    Ok(())
}

fn add_sinusoidal_position(hidden: &mut [f32], position: usize) -> Result<()> {
    if hidden.len() < 4 || hidden.len() % 2 != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "audiocraft sinusoidal position: hidden width {} must be even and >= 4",
            hidden.len()
        )));
    }
    let half = hidden.len() / 2;
    let denominator = (half - 1) as f32;
    let position = position as f32;
    for channel in 0..half {
        let exponent = channel as f32 / denominator;
        let phase = position / POSITION_MAX_PERIOD.powf(exponent);
        hidden[channel] += phase.cos();
        hidden[half + channel] += phase.sin();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    fn bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn add_f32(builder: &mut GgufBuilder, name: &str, shape: &[u64], values: Vec<f32>) {
        let elements = shape.iter().product::<u64>() as usize;
        assert_eq!(values.len(), elements, "fixture shape for {name}");
        builder
            .add_tensor(name, GgmlType::F32, shape.to_vec(), bytes(&values))
            .unwrap();
    }

    fn fixture() -> (Arc<GgufFile>, AudioCraftLmConfig) {
        let config = AudioCraftLmConfig {
            d_model: 4,
            num_layers: 1,
            n_heads: 2,
            ffn_dim: 8,
            vocab_size: 3,
            num_codebooks: 2,
        };
        let (d, ffn, condition_dim) = (config.d_model, config.ffn_dim, 3usize);
        let mut builder = GgufBuilder::new();
        add_f32(
            &mut builder,
            "condition_provider.conditioners.description.output_proj.weight",
            &[d as u64, condition_dim as u64],
            vec![0.0; d * condition_dim],
        );
        add_f32(
            &mut builder,
            "condition_provider.conditioners.description.output_proj.bias",
            &[d as u64],
            vec![1.0; d],
        );
        add_f32(&mut builder, "out_norm.weight", &[d as u64], vec![1.0; d]);
        add_f32(&mut builder, "out_norm.bias", &[d as u64], vec![0.0; d]);
        for codebook in 0..config.num_codebooks {
            let rows = config.vocab_size + 1;
            let mut embedding = vec![0.0; rows * d];
            for row in 0..rows {
                for channel in 0..d {
                    embedding[row * d + channel] = (1 + codebook + row + channel) as f32 * 0.03125;
                }
            }
            add_f32(
                &mut builder,
                &format!("emb.{codebook}.weight"),
                &[rows as u64, d as u64],
                embedding,
            );
            let mut head = vec![0.0; config.vocab_size * d];
            for row in 0..config.vocab_size {
                head[row * d + row] = 1.0 + codebook as f32;
            }
            add_f32(
                &mut builder,
                &format!("linears.{codebook}.weight"),
                &[config.vocab_size as u64, d as u64],
                head,
            );
        }

        let p = "transformer.layers.0";
        for suffix in ["norm1.weight", "norm_cross.weight", "norm2.weight"] {
            add_f32(
                &mut builder,
                &format!("{p}.{suffix}"),
                &[d as u64],
                vec![1.0; d],
            );
        }
        for suffix in ["norm1.bias", "norm_cross.bias", "norm2.bias"] {
            add_f32(
                &mut builder,
                &format!("{p}.{suffix}"),
                &[d as u64],
                vec![0.0; d],
            );
        }
        for prefix in ["self_attn", "cross_attention"] {
            add_f32(
                &mut builder,
                &format!("{p}.{prefix}.in_proj_weight"),
                &[(3 * d) as u64, d as u64],
                vec![0.0; 3 * d * d],
            );
            add_f32(
                &mut builder,
                &format!("{p}.{prefix}.out_proj.weight"),
                &[d as u64, d as u64],
                vec![0.0; d * d],
            );
        }
        add_f32(
            &mut builder,
            &format!("{p}.linear1.weight"),
            &[ffn as u64, d as u64],
            vec![0.0; ffn * d],
        );
        add_f32(
            &mut builder,
            &format!("{p}.linear2.weight"),
            &[d as u64, ffn as u64],
            vec![0.0; d * ffn],
        );

        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        (Arc::new(file), config)
    }

    #[test]
    fn config_rejects_non_head_aligned_and_odd_widths() {
        let mut config = AudioCraftLmConfig {
            d_model: 8,
            num_layers: 1,
            n_heads: 2,
            ffn_dim: 16,
            vocab_size: 8,
            num_codebooks: 4,
        };
        assert!(config.validate().is_ok());
        config.d_model = 7;
        assert!(config.validate().is_err());
        config.d_model = 10;
        config.n_heads = 4;
        assert!(config.validate().is_err());
    }

    #[test]
    fn sinusoidal_position_matches_audiocraft_concat_order() {
        let mut at_zero = vec![0.0; 8];
        add_sinusoidal_position(&mut at_zero, 0).unwrap();
        assert_eq!(at_zero, vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        let mut at_one = vec![0.0; 8];
        add_sinusoidal_position(&mut at_one, 1).unwrap();
        assert_eq!(at_one[0].to_bits(), 1.0f32.cos().to_bits());
        assert_eq!(at_one[4].to_bits(), 1.0f32.sin().to_bits());
        assert!(at_one.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn condition_mask_is_applied_after_the_biased_projection() {
        let (file, config) = fixture();
        let decoder = AudioCraftLmDecoder::bind(file, config, BackendKind::Cpu).unwrap();
        let condition = decoder
            .prepare_condition(&[0.5; 6], 2, Some(&[1, 0]))
            .unwrap();
        assert_eq!(condition.as_slice()[..4], [1.0; 4]);
        assert_eq!(condition.as_slice()[4..], [0.0; 4]);
    }

    #[test]
    fn condition_and_state_cannot_cross_decoder_boundaries() {
        let (first_file, config) = fixture();
        let first = AudioCraftLmDecoder::bind(first_file, config, BackendKind::Cpu).unwrap();
        let condition = first.prepare_condition(&[0.25; 6], 2, None).unwrap();

        let (second_file, second_config) = fixture();
        let second =
            AudioCraftLmDecoder::bind(second_file, second_config, BackendKind::Cpu).unwrap();
        assert!(
            second
                .new_state(&condition, 1)
                .unwrap_err()
                .to_string()
                .contains("different decoder/checkpoint")
        );

        let mut state = first.new_state(&condition, 1).unwrap();
        let mut logits = vec![0.0; config.num_codebooks * config.vocab_size];
        assert!(
            second
                .step_into(&mut state, &[config.special_token_id(); 2], &mut logits)
                .unwrap_err()
                .to_string()
                .contains("different decoder/checkpoint")
        );
    }

    #[test]
    fn mapped_cpu_step_is_deterministic_and_resettable() {
        let (file, config) = fixture();
        let decoder = AudioCraftLmDecoder::bind(file, config, BackendKind::Cpu).unwrap();
        let condition = decoder.prepare_condition(&[0.25; 6], 2, None).unwrap();
        let mut state = decoder.new_state(&condition, 1).unwrap();
        let mut first = vec![0.0; config.num_codebooks * config.vocab_size];
        decoder
            .step_into(&mut state, &[config.special_token_id(); 2], &mut first)
            .unwrap();
        assert_eq!(state.position(), 1);
        assert!(first.iter().all(|value| value.is_finite()));
        assert!(
            decoder
                .step_into(&mut state, &[0, 0], &mut first)
                .unwrap_err()
                .to_string()
                .contains("max_steps")
        );

        state.reset();
        let mut second = vec![0.0; first.len()];
        decoder
            .step_into(&mut state, &[config.special_token_id(); 2], &mut second)
            .unwrap();
        assert_eq!(
            first
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}
