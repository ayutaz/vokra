//! Shared AudioCraft autoregressive LM for the public MusicGen-Medium/Large
//! and AudioGen-Medium GGUF layouts.
//!
//! The public artifacts keep AudioCraft's upstream tensor names verbatim. This
//! module binds that exact LM-only layout, leaves embeddings, output heads and
//! transformer blocks in the GGUF mapping, and widens one layer at a time into
//! a reused scratch block. It deliberately does not tokenize text or decode
//! EnCodec waveform samples: callers supply T5 hidden states and can receive
//! either four codebook-logit rows or frame-major codes after the official
//! delay/CFG/sampling loop. Those two companion boundaries remain explicit.
//!
//! All learned reductions use [`Compute`]. CPU is the reference backend;
//! Metal uses the existing fused attention and MLP paths. Embedding lookup,
//! sinusoidal position construction, residual addition and delay-pattern
//! scheduling are deterministic host-side layout glue, not a hidden CPU model
//! fallback.

use std::sync::{Arc, Mutex, MutexGuard};

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile, GgufTensorInfo};
use vokra_core::{KvCache, Result, Sampler, SamplerConfig, VokraError, apply_cfg_inplace};
use vokra_ops::musicgen_delay_pattern::{
    MUSICGEN_PREDICT_TOKEN, MusicGenDelayPatternAttrs, build_musicgen_delay_pattern,
};

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
/// Checkpointed sinusoidal position-table rows in public Transformers MusicGen.
const TRANSFORMERS_MAX_POSITIONS: usize = 2_048;
/// Checkpointed sinusoidal position-table rows in public Parler-TTS Mini.
const PARLER_MAX_POSITIONS: usize = 4_096;

/// AudioCraft's released generation default.
pub const AUDIOCRAFT_DEFAULT_CFG_COEF: f32 = 3.0;
/// AudioCraft's released sampling temperature.
pub const AUDIOCRAFT_DEFAULT_TEMPERATURE: f32 = 1.0;
/// AudioCraft's released top-k cutoff.
pub const AUDIOCRAFT_DEFAULT_TOP_K: usize = 250;

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

/// Host-side generation controls for an AudioCraft delayed-code sequence.
///
/// `max_frames` counts codec frames before delay interleaving. A positive
/// `temperature` selects seeded sampling. `temperature == 0` is exact greedy
/// argmax. When both `top_p` and `top_k` are present, `top_p` takes precedence,
/// matching AudioCraft's generation branch rather than applying both filters.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioCraftGenerationConfig {
    /// Number of codec frames to return.
    pub max_frames: usize,
    /// Classifier-free guidance coefficient in
    /// `uncond + cfg_coef * (cond - uncond)`.
    pub cfg_coef: f32,
    /// Softmax temperature; zero selects greedy generation.
    pub temperature: f32,
    /// Top-k cutoff used when `top_p` is absent.
    pub top_k: Option<usize>,
    /// Optional nucleus threshold in `(0, 1]`; takes precedence over top-k.
    pub top_p: Option<f32>,
    /// Seed for Vokra's deterministic host sampler.
    pub seed: u64,
}

impl AudioCraftGenerationConfig {
    /// AudioCraft release defaults: CFG 3.0, temperature 1.0 and top-k 250.
    #[must_use]
    pub const fn sampled(max_frames: usize, seed: u64) -> Self {
        Self {
            max_frames,
            cfg_coef: AUDIOCRAFT_DEFAULT_CFG_COEF,
            temperature: AUDIOCRAFT_DEFAULT_TEMPERATURE,
            top_k: Some(AUDIOCRAFT_DEFAULT_TOP_K),
            top_p: None,
            seed,
        }
    }

    /// Deterministic greedy generation with AudioCraft's CFG 3.0 default.
    #[must_use]
    pub const fn greedy(max_frames: usize) -> Self {
        Self {
            max_frames,
            cfg_coef: AUDIOCRAFT_DEFAULT_CFG_COEF,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            seed: 0,
        }
    }

    fn validate(&self, model: AudioCraftLmConfig) -> Result<()> {
        let min_frames = model.num_codebooks.saturating_sub(1).max(1);
        if self.max_frames < min_frames {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft generation: max_frames {} must be >= {min_frames} for the \
                 authenticated complete delay pattern with {} codebooks",
                self.max_frames, model.num_codebooks
            )));
        }
        if !self.cfg_coef.is_finite() || self.cfg_coef < 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft generation: cfg_coef must be finite and >= 0, got {}",
                self.cfg_coef
            )));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft generation: temperature must be finite and >= 0, got {}",
                self.temperature
            )));
        }
        if self.top_p.is_none() {
            if let Some(top_k) = self.top_k {
                if top_k == 0 || top_k > model.vocab_size {
                    return Err(VokraError::InvalidArgument(format!(
                        "audiocraft generation: top_k {top_k} must be in 1..={}",
                        model.vocab_size
                    )));
                }
            }
        }
        if let Some(top_p) = self.top_p {
            if !(top_p.is_finite() && 0.0 < top_p && top_p <= 1.0) {
                return Err(VokraError::InvalidArgument(format!(
                    "audiocraft generation: top_p must be finite and in (0, 1], got {top_p}"
                )));
            }
        }
        Ok(())
    }

    fn sampler_config(&self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.temperature,
            top_k: if self.top_p.is_some() {
                None
            } else {
                self.top_k
            },
            top_p: self.top_p,
            repetition_penalty: None,
            seed: self.seed,
        }
    }
}

/// Generated EnCodec indices in `[frame, codebook]` row-major order.
///
/// This is the layout consumed directly by
/// [`vokra_ops::encodec_rvq::encodec_rvq_decode`]. Every stored id is strictly
/// below the LM vocabulary; delay-padding/special tokens are removed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioCraftGeneratedCodes {
    codes: Vec<u32>,
    frames: usize,
    num_codebooks: usize,
}

impl AudioCraftGeneratedCodes {
    /// Returns the number of generated codec frames.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    /// Returns the number of codec codebooks per frame.
    #[must_use]
    pub const fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    /// Frame-major `[frames, num_codebooks]` indices.
    #[must_use]
    pub fn as_frame_major(&self) -> &[u32] {
        &self.codes
    }

    /// Consumes the result and returns frame-major indices.
    #[must_use]
    pub fn into_frame_major(self) -> Vec<u32> {
        self.codes
    }

    /// One frame's contiguous codebook row.
    pub fn frame(&self, frame: usize) -> Result<&[u32]> {
        if frame >= self.frames {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft generated codes: frame {frame} >= {}",
                self.frames
            )));
        }
        let start = frame * self.num_codebooks;
        Ok(&self.codes[start..start + self.num_codebooks])
    }
}

/// Shape contract for one AudioCraft LM-only checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioCraftLmConfig {
    /// Transformer hidden width.
    pub d_model: usize,
    /// Number of transformer decoder layers.
    pub num_layers: usize,
    /// Number of attention heads per layer.
    pub n_heads: usize,
    /// Feed-forward width per layer.
    pub ffn_dim: usize,
    /// Codec-token vocabulary size, excluding the special token.
    pub vocab_size: usize,
    /// Number of independently predicted codec codebooks.
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

    /// Returns the reserved delay-padding and terminal token id.
    #[must_use]
    pub const fn special_token_id(self) -> u32 {
        self.vocab_size as u32
    }
}

/// T5 condition after the layout's projection (or Parler identity) and mask.
#[derive(Debug, Clone)]
pub struct AudioCraftCondition {
    projected: Vec<f32>,
    frames: usize,
    d_model: usize,
    decoder_identity: Arc<()>,
}

impl AudioCraftCondition {
    /// Returns the number of visible text-condition frames.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    /// Returns the projected condition width.
    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.d_model
    }

    /// Returns the row-major projected condition values.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.projected
    }
}

enum MappedAttentionLocs {
    Fused {
        in_proj: GgufTensorInfo,
    },
    Split {
        q: GgufTensorInfo,
        k: GgufTensorInfo,
        v: GgufTensorInfo,
    },
}

struct MappedLayerLocs {
    self_attn: MappedAttentionLocs,
    self_out: GgufTensorInfo,
    cross_attn: MappedAttentionLocs,
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
    linears: MappedHeadLocs,
    chunk: Mutex<Vec<f32>>,
}

enum MappedHeadLocs {
    Separate(Vec<GgufTensorInfo>),
    /// One row-major `[num_codebooks * vocab_size, d_model]` tensor. Parler's
    /// multilingual checkpoint fuses the otherwise independent codebook
    /// heads without changing their row order.
    Fused(GgufTensorInfo),
}

enum MappedPosition {
    AudioCraftSinusoidal,
    TransformersTable(GgufTensorInfo),
}

#[derive(Clone, Copy)]
enum ConditionMaskMode {
    /// AudioCraft's conditioner keeps the sequence length and zeros masked rows.
    ZeroRows,
    /// Transformers supplies an attention mask. For the batch-one native API,
    /// removing masked rows before K/V projection is exactly equivalent.
    CompactRows,
}

enum ConditionProjection {
    Linear(Linear),
    /// Parler's embedded FLAN-T5 width already equals the decoder width, so
    /// Transformers does not construct `enc_to_dec_proj` at all.
    Identity,
}

struct MappedWeights {
    file: Arc<GgufFile>,
    layers: Vec<MappedLayerLocs>,
    layer_scratch: Mutex<DecoderLayer>,
    heads: MappedHeads,
    position: MappedPosition,
    condition_mask_mode: ConditionMaskMode,
    condition_proj: ConditionProjection,
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
                self_attn: MappedAttentionLocs::Fused {
                    in_proj: exact_info(
                        &file,
                        &format!("{p}.self_attn.in_proj_weight"),
                        &[3 * d, d],
                    )?,
                },
                self_out: exact_info(&file, &format!("{p}.self_attn.out_proj.weight"), &[d, d])?,
                cross_attn: MappedAttentionLocs::Fused {
                    in_proj: exact_info(
                        &file,
                        &format!("{p}.cross_attention.in_proj_weight"),
                        &[3 * d, d],
                    )?,
                },
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
                    linears: MappedHeadLocs::Separate(linears),
                    chunk: Mutex::new(Vec::new()),
                },
                position: MappedPosition::AudioCraftSinusoidal,
                condition_mask_mode: ConditionMaskMode::ZeroRows,
                condition_proj: ConditionProjection::Linear(Linear::dense(
                    condition_w_t,
                    condition_dim,
                    d,
                    Some(condition_b),
                )),
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

    /// Binds the Transformers-composite decoder layout used by the public
    /// MusicGen Small/Melody artifacts.
    ///
    /// Its math is the same pre-norm self-attention, cross-attention and GELU
    /// MLP stack as the AudioCraft layout, but Q/K/V tensors are split,
    /// positions come from the checkpoint's fixed table and T5 projection
    /// names live at the composite root. Keeping this as an authenticated
    /// layout entry avoids copying the 24/48-layer checkpoint into memory.
    pub(crate) fn bind_transformers_musicgen(
        file: Arc<GgufFile>,
        config: AudioCraftLmConfig,
        backend: BackendKind,
    ) -> Result<Self> {
        config.validate()?;
        Compute::for_backend(backend, AUDIOCRAFT_LM_HOT_OPS)?;
        let d = config.d_model;
        let ffn = config.ffn_dim;
        let rows = config.vocab_size + 1;

        let condition_weight_name = "enc_to_dec_proj.weight";
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
        let condition_bias = exact_info(&file, "enc_to_dec_proj.bias", &[d])?;
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

        let decoder = "decoder.model.decoder";
        let out_norm_w = exact_info(&file, &format!("{decoder}.layer_norm.weight"), &[d])?;
        let out_norm_b = exact_info(&file, &format!("{decoder}.layer_norm.bias"), &[d])?;
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
                &format!("{decoder}.embed_tokens.{codebook}.weight"),
                &[rows, d],
            )?);
            linears.push(exact_info(
                &file,
                &format!("decoder.lm_heads.{codebook}.weight"),
                &[config.vocab_size, d],
            )?);
        }
        let position = exact_info(
            &file,
            &format!("{decoder}.embed_positions.weights"),
            &[TRANSFORMERS_MAX_POSITIONS, d],
        )?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let p = format!("{decoder}.layers.{layer}");
            let split = |name: &str| -> Result<MappedAttentionLocs> {
                Ok(MappedAttentionLocs::Split {
                    q: exact_info(&file, &format!("{name}.q_proj.weight"), &[d, d])?,
                    k: exact_info(&file, &format!("{name}.k_proj.weight"), &[d, d])?,
                    v: exact_info(&file, &format!("{name}.v_proj.weight"), &[d, d])?,
                })
            };
            layers.push(MappedLayerLocs {
                self_attn: split(&format!("{p}.self_attn"))?,
                self_out: exact_info(&file, &format!("{p}.self_attn.out_proj.weight"), &[d, d])?,
                cross_attn: split(&format!("{p}.encoder_attn"))?,
                cross_out: exact_info(
                    &file,
                    &format!("{p}.encoder_attn.out_proj.weight"),
                    &[d, d],
                )?,
                norm1_w: exact_info(&file, &format!("{p}.self_attn_layer_norm.weight"), &[d])?,
                norm1_b: exact_info(&file, &format!("{p}.self_attn_layer_norm.bias"), &[d])?,
                norm_cross_w: exact_info(
                    &file,
                    &format!("{p}.encoder_attn_layer_norm.weight"),
                    &[d],
                )?,
                norm_cross_b: exact_info(
                    &file,
                    &format!("{p}.encoder_attn_layer_norm.bias"),
                    &[d],
                )?,
                norm2_w: exact_info(&file, &format!("{p}.final_layer_norm.weight"), &[d])?,
                norm2_b: exact_info(&file, &format!("{p}.final_layer_norm.bias"), &[d])?,
                fc1: exact_info(&file, &format!("{p}.fc1.weight"), &[ffn, d])?,
                fc2: exact_info(&file, &format!("{p}.fc2.weight"), &[d, ffn])?,
            });
        }

        Ok(Self {
            weights: MappedWeights {
                file,
                layers,
                layer_scratch: Mutex::new(empty_layer(d, ffn)),
                heads: MappedHeads {
                    embeddings,
                    linears: MappedHeadLocs::Separate(linears),
                    chunk: Mutex::new(Vec::new()),
                },
                position: MappedPosition::TransformersTable(position),
                condition_mask_mode: ConditionMaskMode::CompactRows,
                condition_proj: ConditionProjection::Linear(Linear::dense(
                    condition_w_t,
                    condition_dim,
                    d,
                    Some(condition_b),
                )),
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

    /// Binds the Transformers decoder layout embedded in public Parler-TTS
    /// Mini checkpoints.
    ///
    /// Parler shares MusicGen's split-QKV decoder math but has no
    /// `enc_to_dec_proj` because FLAN-T5-large and the decoder are both width
    /// 1024. Its position table has 4096 rows, and multilingual v1.1 stores
    /// the nine output heads as one row-concatenated tensor. The caller
    /// authenticates the complete model manifest before entering this
    /// layout-specific binder.
    pub(crate) fn bind_transformers_parler(
        file: Arc<GgufFile>,
        config: AudioCraftLmConfig,
        backend: BackendKind,
        fused_lm_heads: bool,
    ) -> Result<Self> {
        config.validate()?;
        Compute::for_backend(backend, AUDIOCRAFT_LM_HOT_OPS)?;
        let d = config.d_model;
        let ffn = config.ffn_dim;
        let rows = config.vocab_size + 1;
        let decoder = "decoder.model.decoder";

        let out_norm_w = exact_info(&file, &format!("{decoder}.layer_norm.weight"), &[d])?;
        let out_norm_b = exact_info(&file, &format!("{decoder}.layer_norm.bias"), &[d])?;
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
        for codebook in 0..config.num_codebooks {
            embeddings.push(exact_info(
                &file,
                &format!("{decoder}.embed_tokens.{codebook}.weight"),
                &[rows, d],
            )?);
        }
        let linears = if fused_lm_heads {
            let fused_rows = config
                .num_codebooks
                .checked_mul(config.vocab_size)
                .ok_or_else(|| {
                    VokraError::ModelLoad(
                        "audiocraft-lm: Parler fused LM-head rows overflow usize".to_owned(),
                    )
                })?;
            MappedHeadLocs::Fused(exact_info(
                &file,
                "decoder.lm_heads.weight",
                &[fused_rows, d],
            )?)
        } else {
            let mut linears = Vec::with_capacity(config.num_codebooks);
            for codebook in 0..config.num_codebooks {
                linears.push(exact_info(
                    &file,
                    &format!("decoder.lm_heads.{codebook}.weight"),
                    &[config.vocab_size, d],
                )?);
            }
            MappedHeadLocs::Separate(linears)
        };
        let position = exact_info(
            &file,
            &format!("{decoder}.embed_positions.weights"),
            &[PARLER_MAX_POSITIONS, d],
        )?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let p = format!("{decoder}.layers.{layer}");
            let split = |name: &str| -> Result<MappedAttentionLocs> {
                Ok(MappedAttentionLocs::Split {
                    q: exact_info(&file, &format!("{name}.q_proj.weight"), &[d, d])?,
                    k: exact_info(&file, &format!("{name}.k_proj.weight"), &[d, d])?,
                    v: exact_info(&file, &format!("{name}.v_proj.weight"), &[d, d])?,
                })
            };
            layers.push(MappedLayerLocs {
                self_attn: split(&format!("{p}.self_attn"))?,
                self_out: exact_info(&file, &format!("{p}.self_attn.out_proj.weight"), &[d, d])?,
                cross_attn: split(&format!("{p}.encoder_attn"))?,
                cross_out: exact_info(
                    &file,
                    &format!("{p}.encoder_attn.out_proj.weight"),
                    &[d, d],
                )?,
                norm1_w: exact_info(&file, &format!("{p}.self_attn_layer_norm.weight"), &[d])?,
                norm1_b: exact_info(&file, &format!("{p}.self_attn_layer_norm.bias"), &[d])?,
                norm_cross_w: exact_info(
                    &file,
                    &format!("{p}.encoder_attn_layer_norm.weight"),
                    &[d],
                )?,
                norm_cross_b: exact_info(
                    &file,
                    &format!("{p}.encoder_attn_layer_norm.bias"),
                    &[d],
                )?,
                norm2_w: exact_info(&file, &format!("{p}.final_layer_norm.weight"), &[d])?,
                norm2_b: exact_info(&file, &format!("{p}.final_layer_norm.bias"), &[d])?,
                fc1: exact_info(&file, &format!("{p}.fc1.weight"), &[ffn, d])?,
                fc2: exact_info(&file, &format!("{p}.fc2.weight"), &[d, ffn])?,
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
                position: MappedPosition::TransformersTable(position),
                condition_mask_mode: ConditionMaskMode::CompactRows,
                condition_proj: ConditionProjection::Identity,
                out_norm: LayerNorm {
                    gamma: out_gamma,
                    beta: out_beta,
                },
                condition_dim: d,
                config,
            },
            backend,
            identity: Arc::new(()),
        })
    }

    /// Returns the authenticated language-model topology.
    #[must_use]
    pub const fn config(&self) -> AudioCraftLmConfig {
        self.weights.config
    }

    /// Returns the text-condition width expected by this checkpoint layout.
    #[must_use]
    pub const fn condition_dim(&self) -> usize {
        self.weights.condition_dim
    }

    /// Returns the selected execution backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Applies the layout's T5 output projection and upstream mask.
    ///
    /// `hidden` is `[frames, condition_dim]`. `mask`, when provided, contains
    /// exactly `frames` values in `{0,1}`. AudioCraft applies the linear first
    /// and then zeros masked rows. Transformers passes an attention mask; this
    /// batch-one route preserves that math by compacting visible rows before
    /// cross-attention K/V projection. MusicGen's biased projection precedes
    /// compaction; Parler selects the strict width-preserving identity path.
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
        let mut projected = match &self.weights.condition_proj {
            ConditionProjection::Linear(projection) => {
                let mut projected = vec![0.0; frames * d];
                let compute = self.compute()?;
                crate::whisper::nn::linear_apply(
                    &compute,
                    &mut projected,
                    hidden,
                    frames,
                    projection,
                )?;
                projected
            }
            ConditionProjection::Identity => hidden.to_vec(),
        };
        let frames = match (self.weights.condition_mask_mode, mask) {
            (ConditionMaskMode::ZeroRows, Some(mask)) => {
                for (frame, &visible) in mask.iter().enumerate() {
                    if visible == 0 {
                        projected[frame * d..(frame + 1) * d].fill(0.0);
                    }
                }
                frames
            }
            (ConditionMaskMode::CompactRows, Some(mask)) => {
                let visible_frames = mask.iter().filter(|&&visible| visible == 1).count();
                if visible_frames == 0 {
                    return Err(VokraError::InvalidArgument(
                        "audiocraft condition: Transformers attention mask hides every frame"
                            .to_owned(),
                    ));
                }
                if visible_frames != frames {
                    let mut compact = Vec::with_capacity(visible_frames * d);
                    for (frame, &visible) in mask.iter().enumerate() {
                        if visible == 1 {
                            compact.extend_from_slice(&projected[frame * d..(frame + 1) * d]);
                        }
                    }
                    projected = compact;
                }
                visible_frames
            }
            (_, None) => frames,
        };
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
        if let MappedPosition::TransformersTable(info) = &self.weights.position {
            let position_rows = dims(info)[0];
            if max_steps > position_rows {
                return Err(VokraError::InvalidArgument(format!(
                    "audiocraft LM state: max_steps {max_steps} exceeds Transformers \
                     position table rows {position_rows} (no silent wrap)"
                )));
            }
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

    /// Creates a state and causally prefills arbitrary decoder embeddings
    /// before the first codebook-token step.
    ///
    /// Parler-TTS uses this for `embed_prompts(prompt_input_ids)`: prompt rows
    /// occupy positions `0..prompt_len`, populate every self-attention cache,
    /// and the first delayed BOS tuple starts at `prompt_len`. The prefix is
    /// not projected through the codebook embedding tables and never emits LM
    /// logits. `max_decode_steps` counts only subsequent codebook-token steps.
    pub(crate) fn new_state_with_prefix_embeddings(
        &self,
        condition: &AudioCraftCondition,
        prefix_embeddings: &[f32],
        max_decode_steps: usize,
    ) -> Result<AudioCraftLmState> {
        let d = self.weights.config.d_model;
        if prefix_embeddings.len() % d != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft LM prefix len {} is not divisible by d_model {d}",
                prefix_embeddings.len()
            )));
        }
        if prefix_embeddings.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "audiocraft LM prefix contains a non-finite value".to_owned(),
            ));
        }
        let prefix_frames = prefix_embeddings.len() / d;
        let total_steps = prefix_frames.checked_add(max_decode_steps).ok_or_else(|| {
            VokraError::InvalidArgument(
                "audiocraft LM prefix + decode steps overflow usize".to_owned(),
            )
        })?;
        let mut state = self.new_state(condition, total_steps)?;
        for row in prefix_embeddings.chunks_exact(d) {
            state.h.copy_from_slice(row);
            self.advance_hidden(&mut state, None)?;
        }
        Ok(state)
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
        self.advance_hidden(state, Some(logits))
    }

    fn advance_hidden(
        &self,
        state: &mut AudioCraftLmState,
        logits: Option<&mut [f32]>,
    ) -> Result<()> {
        let cfg = self.weights.config;
        state.validate_for(cfg, &self.identity)?;
        let start = state.self_kv.positions();
        if start >= state.max_steps {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft LM step: position {start} reached max_steps {} (no silent wrap)",
                state.max_steps
            )));
        }
        match &self.weights.position {
            MappedPosition::AudioCraftSinusoidal => {
                add_sinusoidal_position(&mut state.h, start)?;
            }
            MappedPosition::TransformersTable(info) => {
                add_mapped_row(&self.weights.file, info, start, cfg.d_model, &mut state.h)?;
            }
        }

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
        if let Some(logits) = logits {
            layer_norm_into(
                &compute,
                &mut state.block.ln,
                &state.h,
                1,
                &self.weights.out_norm,
            )?;
            self.logits_into(&compute, &state.block.ln, logits)?;
        }
        state.self_kv.advance(1);
        state.poisoned = false;
        Ok(())
    }

    /// Generates raw RVQ indices from prepared conditional and null conditions.
    ///
    /// This is AudioCraft's memory-lean two-step CFG route: conditional and
    /// unconditional KV states advance over the same delayed token sequence,
    /// their logits are combined with [`apply_cfg_inplace`], and a single
    /// seeded sampler draws every codebook row before the delay mask is
    /// applied. The returned codes are frame-major and contain no special
    /// token. Text tokenization and EnCodec waveform decoding remain separate
    /// authenticated companion boundaries.
    pub fn generate_codes(
        &self,
        conditional: &AudioCraftCondition,
        unconditional: &AudioCraftCondition,
        generation: &AudioCraftGenerationConfig,
    ) -> Result<AudioCraftGeneratedCodes> {
        let cfg = self.weights.config;
        generation.validate(cfg)?;

        let max_length = generation
            .max_frames
            .checked_add(cfg.num_codebooks)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "audiocraft generation: max_frames + num_codebooks overflows usize".to_owned(),
                )
            })?;
        let delayed_steps = max_length - 1;
        let special = cfg.special_token_id();
        let prefix = vec![special; cfg.num_codebooks];
        let delay = build_musicgen_delay_pattern(
            &prefix,
            MusicGenDelayPatternAttrs {
                batch_size: 1,
                num_codebooks: cfg.num_codebooks,
                prompt_len: 1,
                max_length,
                audio_channels: 1,
                pad_token_id: special,
            },
        )?;
        if delay.rows != cfg.num_codebooks
            || delay.prefix_len != 1
            || delay.prefix.len() != cfg.num_codebooks
        {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft generation: delay prefix shape [{}, {}] does not match [{}, 1]",
                delay.rows, delay.prefix_len, cfg.num_codebooks
            )));
        }

        let mut conditional_state = self.new_state(conditional, delayed_steps)?;
        let mut unconditional_state = self.new_state(unconditional, delayed_steps)?;
        let logits_len = cfg
            .num_codebooks
            .checked_mul(cfg.vocab_size)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "audiocraft generation: logits shape overflows usize".to_owned(),
                )
            })?;
        let sequence_len = cfg.num_codebooks.checked_mul(max_length).ok_or_else(|| {
            VokraError::InvalidArgument(
                "audiocraft generation: delayed sequence shape overflows usize".to_owned(),
            )
        })?;
        let mut conditional_logits = vec![0.0; logits_len];
        let mut unconditional_logits = vec![0.0; logits_len];
        let mut sequence = vec![special; sequence_len];
        let mut current_tokens = delay.prefix.clone();
        for codebook in 0..cfg.num_codebooks {
            sequence[codebook * max_length] = delay.prefix[codebook];
        }
        let mut sampler = Sampler::new(generation.sampler_config());

        for target_offset in 1..max_length {
            self.step_into(
                &mut conditional_state,
                &current_tokens,
                &mut conditional_logits,
            )?;
            self.step_into(
                &mut unconditional_state,
                &current_tokens,
                &mut unconditional_logits,
            )?;
            apply_cfg_inplace(
                &mut conditional_logits,
                &unconditional_logits,
                generation.cfg_coef,
            )?;
            if conditional_logits.iter().any(|value| !value.is_finite()) {
                return Err(VokraError::InvalidArgument(format!(
                    "audiocraft generation: non-finite guided logit at delayed position {target_offset}"
                )));
            }

            for (codebook, current) in current_tokens.iter_mut().enumerate() {
                let logits = &mut conditional_logits
                    [codebook * cfg.vocab_size..(codebook + 1) * cfg.vocab_size];
                // AudioCraft samples all rows first, then forces the rows that
                // are invalid at this delayed position back to the special id.
                let sampled = sampler.sample(logits);
                let mask_token = delay.pattern[codebook * max_length + target_offset];
                let token = if mask_token == MUSICGEN_PREDICT_TOKEN {
                    sampled
                } else {
                    u32::try_from(mask_token).map_err(|_| {
                        VokraError::InvalidArgument(format!(
                            "audiocraft generation: delay token {mask_token} at codebook \
                             {codebook}, position {target_offset} does not fit u32"
                        ))
                    })?
                };
                sequence[codebook * max_length + target_offset] = token;
                *current = token;
            }
        }

        let output_len = generation
            .max_frames
            .checked_mul(cfg.num_codebooks)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "audiocraft generation: output code shape overflows usize".to_owned(),
                )
            })?;
        let mut codes = vec![0; output_len];
        for frame in 0..generation.max_frames {
            for codebook in 0..cfg.num_codebooks {
                let delayed_position = 1 + codebook + frame;
                let pattern_index = codebook * max_length + delayed_position;
                if delay.pattern[pattern_index] != MUSICGEN_PREDICT_TOKEN {
                    return Err(VokraError::InvalidArgument(format!(
                        "audiocraft generation: expected a predictive delay slot at frame \
                         {frame}, codebook {codebook}, position {delayed_position}"
                    )));
                }
                let token = sequence[pattern_index];
                if token as usize >= cfg.vocab_size {
                    return Err(VokraError::InvalidArgument(format!(
                        "audiocraft generation: extracted special/out-of-range token {token} at \
                         frame {frame}, codebook {codebook}"
                    )));
                }
                codes[frame * cfg.num_codebooks + codebook] = token;
            }
        }

        Ok(AudioCraftGeneratedCodes {
            codes,
            frames: generation.max_frames,
            num_codebooks: cfg.num_codebooks,
        })
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
        materialize_attention(
            f,
            &locs.self_attn,
            &locs.self_out,
            d,
            &mut scratch.self_attn,
        )?;
        materialize_norm(
            f,
            &locs.norm_cross_w,
            &locs.norm_cross_b,
            &mut scratch.cross_ln,
        )?;
        materialize_attention(
            f,
            &locs.cross_attn,
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
        for codebook in 0..cfg.num_codebooks {
            let (info, head_base) = match &self.weights.heads.linears {
                MappedHeadLocs::Separate(linears) => (&linears[codebook], 0),
                MappedHeadLocs::Fused(info) => {
                    let head_elements = vocab.checked_mul(d).ok_or_else(|| {
                        VokraError::InvalidArgument(
                            "audiocraft LM fused head shape overflows usize".to_owned(),
                        )
                    })?;
                    let head_bytes = head_elements
                        .checked_mul(info.dtype.type_size())
                        .ok_or_else(|| {
                            VokraError::InvalidArgument(
                                "audiocraft LM fused head byte offset overflows usize".to_owned(),
                            )
                        })?;
                    (
                        info,
                        codebook.checked_mul(head_bytes).ok_or_else(|| {
                            VokraError::InvalidArgument(
                                "audiocraft LM fused codebook offset overflows usize".to_owned(),
                            )
                        })?,
                    )
                }
            };
            let bytes = self.weights.file.tensor_bytes(info);
            let esz = info.dtype.type_size();
            let codebook_out = &mut out[codebook * vocab..(codebook + 1) * vocab];
            let mut row = 0usize;
            while row < vocab {
                let rows = HEAD_CHUNK_ROWS.min(vocab - row);
                let n = rows * d;
                let start = head_base + row * d * esz;
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
    /// Returns the number of autoregressive positions already evaluated.
    #[must_use]
    pub fn position(&self) -> usize {
        self.self_kv.positions()
    }

    /// Returns the maximum number of autoregressive steps allocated.
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
    input: &MappedAttentionLocs,
    out_proj: &GgufTensorInfo,
    d: usize,
    out: &mut Attention,
) -> Result<()> {
    match input {
        MappedAttentionLocs::Fused { in_proj } => {
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
        }
        MappedAttentionLocs::Split { q, k, v } => {
            for (info, linear) in [(q, &mut out.q), (k, &mut out.k), (v, &mut out.v)] {
                transpose_widen(
                    file.tensor_bytes(info),
                    info.dtype,
                    d,
                    d,
                    dense_weight_mut(linear)?,
                    AUDIOCRAFT_MAPPED,
                )?;
                linear.bias = None;
            }
        }
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

    fn transformers_fixture_for(
        parler: bool,
        fused_lm_heads: bool,
    ) -> (Arc<GgufFile>, AudioCraftLmConfig) {
        let config = AudioCraftLmConfig {
            d_model: 4,
            num_layers: 1,
            n_heads: 2,
            ffn_dim: 8,
            vocab_size: 3,
            num_codebooks: 2,
        };
        let (d, ffn) = (config.d_model, config.ffn_dim);
        let condition_dim = if parler { d } else { 3usize };
        let decoder = "decoder.model.decoder";
        let mut builder = GgufBuilder::new();
        if !parler {
            add_f32(
                &mut builder,
                "enc_to_dec_proj.weight",
                &[d as u64, condition_dim as u64],
                vec![0.0; d * condition_dim],
            );
            add_f32(
                &mut builder,
                "enc_to_dec_proj.bias",
                &[d as u64],
                vec![1.0; d],
            );
        }
        add_f32(
            &mut builder,
            &format!("{decoder}.layer_norm.weight"),
            &[d as u64],
            vec![1.0; d],
        );
        add_f32(
            &mut builder,
            &format!("{decoder}.layer_norm.bias"),
            &[d as u64],
            vec![0.0; d],
        );
        for codebook in 0..config.num_codebooks {
            let rows = config.vocab_size + 1;
            add_f32(
                &mut builder,
                &format!("{decoder}.embed_tokens.{codebook}.weight"),
                &[rows as u64, d as u64],
                vec![0.125; rows * d],
            );
            if !fused_lm_heads {
                add_f32(
                    &mut builder,
                    &format!("decoder.lm_heads.{codebook}.weight"),
                    &[config.vocab_size as u64, d as u64],
                    vec![0.25; config.vocab_size * d],
                );
            }
        }
        if fused_lm_heads {
            add_f32(
                &mut builder,
                "decoder.lm_heads.weight",
                &[(config.num_codebooks * config.vocab_size) as u64, d as u64],
                vec![0.25; config.num_codebooks * config.vocab_size * d],
            );
        }
        let position_rows = if parler {
            PARLER_MAX_POSITIONS
        } else {
            TRANSFORMERS_MAX_POSITIONS
        };
        let mut positions = vec![0.0; position_rows * d];
        add_sinusoidal_position(&mut positions[..d], 0).unwrap();
        add_sinusoidal_position(&mut positions[d..2 * d], 1).unwrap();
        add_f32(
            &mut builder,
            &format!("{decoder}.embed_positions.weights"),
            &[position_rows as u64, d as u64],
            positions,
        );

        let p = format!("{decoder}.layers.0");
        for suffix in [
            "self_attn_layer_norm.weight",
            "encoder_attn_layer_norm.weight",
            "final_layer_norm.weight",
        ] {
            add_f32(
                &mut builder,
                &format!("{p}.{suffix}"),
                &[d as u64],
                vec![1.0; d],
            );
        }
        for suffix in [
            "self_attn_layer_norm.bias",
            "encoder_attn_layer_norm.bias",
            "final_layer_norm.bias",
        ] {
            add_f32(
                &mut builder,
                &format!("{p}.{suffix}"),
                &[d as u64],
                vec![0.0; d],
            );
        }
        for attention in ["self_attn", "encoder_attn"] {
            for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                add_f32(
                    &mut builder,
                    &format!("{p}.{attention}.{projection}.weight"),
                    &[d as u64, d as u64],
                    vec![0.0; d * d],
                );
            }
        }
        add_f32(
            &mut builder,
            &format!("{p}.fc1.weight"),
            &[ffn as u64, d as u64],
            vec![0.0; ffn * d],
        );
        add_f32(
            &mut builder,
            &format!("{p}.fc2.weight"),
            &[d as u64, ffn as u64],
            vec![0.0; d * ffn],
        );

        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        (Arc::new(file), config)
    }

    fn transformers_fixture() -> (Arc<GgufFile>, AudioCraftLmConfig) {
        transformers_fixture_for(false, false)
    }

    fn parler_fixture(fused_lm_heads: bool) -> (Arc<GgufFile>, AudioCraftLmConfig) {
        transformers_fixture_for(true, fused_lm_heads)
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
    fn transformers_layout_compacts_masked_condition_rows_and_executes_split_attention() {
        let (file, config) = transformers_fixture();
        let decoder =
            AudioCraftLmDecoder::bind_transformers_musicgen(file, config, BackendKind::Cpu)
                .unwrap();
        let condition = decoder
            .prepare_condition(&[0.5; 6], 2, Some(&[1, 0]))
            .unwrap();
        assert_eq!(condition.frames(), 1);
        assert_eq!(condition.as_slice(), [1.0; 4]);
        assert!(
            decoder
                .prepare_condition(&[0.5; 6], 2, Some(&[0, 0]))
                .unwrap_err()
                .to_string()
                .contains("hides every frame")
        );
        assert!(
            decoder
                .new_state(&condition, TRANSFORMERS_MAX_POSITIONS + 1)
                .unwrap_err()
                .to_string()
                .contains("position table rows")
        );

        let mut state = decoder.new_state(&condition, 1).unwrap();
        let mut logits = vec![0.0; config.num_codebooks * config.vocab_size];
        decoder
            .step_into(&mut state, &[config.special_token_id(); 2], &mut logits)
            .unwrap();
        assert!(logits.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn parler_layout_uses_identity_condition_and_supports_both_head_layouts() {
        let mut outputs = Vec::new();
        for fused in [false, true] {
            let (file, config) = parler_fixture(fused);
            let decoder = AudioCraftLmDecoder::bind_transformers_parler(
                file,
                config,
                BackendKind::Cpu,
                fused,
            )
            .unwrap();
            let condition = decoder
                .prepare_condition(&[0.5; 8], 2, Some(&[1, 0]))
                .unwrap();
            assert_eq!(condition.frames(), 1);
            assert_eq!(condition.as_slice(), [0.5; 4]);
            assert!(
                decoder
                    .new_state(&condition, PARLER_MAX_POSITIONS + 1)
                    .unwrap_err()
                    .to_string()
                    .contains("position table rows")
            );

            let mut state = decoder.new_state(&condition, 1).unwrap();
            let mut logits = vec![0.0; config.num_codebooks * config.vocab_size];
            decoder
                .step_into(&mut state, &[config.special_token_id(); 2], &mut logits)
                .unwrap();
            assert!(logits.iter().all(|value| value.is_finite()));
            outputs.push(logits);
        }
        assert_eq!(outputs[0], outputs[1]);
    }

    #[test]
    fn parler_head_layout_mismatch_is_an_explicit_bind_error() {
        let (separate, config) = parler_fixture(false);
        assert!(
            AudioCraftLmDecoder::bind_transformers_parler(
                separate,
                config,
                BackendKind::Cpu,
                true,
            )
            .unwrap_err()
            .to_string()
            .contains("decoder.lm_heads.weight")
        );

        let (fused, config) = parler_fixture(true);
        assert!(
            AudioCraftLmDecoder::bind_transformers_parler(fused, config, BackendKind::Cpu, false,)
                .unwrap_err()
                .to_string()
                .contains("decoder.lm_heads.0.weight")
        );
    }

    #[test]
    fn parler_prompt_embeddings_prefill_self_attention_positions() {
        let (file, config) = parler_fixture(true);
        let decoder =
            AudioCraftLmDecoder::bind_transformers_parler(file, config, BackendKind::Cpu, true)
                .unwrap();
        let condition = decoder.prepare_condition(&[0.5; 4], 1, None).unwrap();
        let mut state = decoder
            .new_state_with_prefix_embeddings(&condition, &[0.125; 8], 1)
            .unwrap();
        assert_eq!(state.position(), 2);
        assert_eq!(state.max_steps(), 3);

        let mut logits = vec![0.0; config.num_codebooks * config.vocab_size];
        decoder
            .step_into(&mut state, &[config.special_token_id(); 2], &mut logits)
            .unwrap();
        assert_eq!(state.position(), 3);
        assert!(logits.iter().all(|value| value.is_finite()));

        assert!(
            decoder
                .new_state_with_prefix_embeddings(&condition, &[0.0; 5], 1)
                .unwrap_err()
                .to_string()
                .contains("not divisible")
        );
        assert!(
            decoder
                .new_state_with_prefix_embeddings(&condition, &[f32::NAN; 4], 1)
                .unwrap_err()
                .to_string()
                .contains("non-finite")
        );
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
    fn generation_config_pins_defaults_and_rejects_incomplete_delay_geometry() {
        let model = AudioCraftLmConfig {
            d_model: 8,
            num_layers: 1,
            n_heads: 2,
            ffn_dim: 16,
            vocab_size: 2_048,
            num_codebooks: 4,
        };
        let defaults = AudioCraftGenerationConfig::sampled(50, 17);
        assert_eq!(defaults.cfg_coef, 3.0);
        assert_eq!(defaults.temperature, 1.0);
        assert_eq!(defaults.top_k, Some(250));
        assert!(defaults.validate(model).is_ok());

        let too_short = AudioCraftGenerationConfig::greedy(2);
        assert!(
            too_short
                .validate(model)
                .unwrap_err()
                .to_string()
                .contains("max_frames")
        );

        let mut nucleus = defaults;
        nucleus.top_p = Some(0.95);
        nucleus.top_k = Some(0); // ignored because AudioCraft gives top-p precedence
        assert!(nucleus.validate(model).is_ok());
        assert_eq!(nucleus.sampler_config().top_k, None);
        assert_eq!(nucleus.sampler_config().top_p, Some(0.95));
    }

    #[test]
    fn generated_codes_are_seeded_frame_major_and_strip_delay_tokens() {
        let (file, config) = fixture();
        let decoder = AudioCraftLmDecoder::bind(file, config, BackendKind::Cpu).unwrap();
        let condition = decoder.prepare_condition(&[0.25; 6], 2, None).unwrap();
        let mut generation = AudioCraftGenerationConfig::sampled(3, 0x1234);
        generation.top_k = Some(config.vocab_size);

        let first = decoder
            .generate_codes(&condition, &condition, &generation)
            .unwrap();
        let second = decoder
            .generate_codes(&condition, &condition, &generation)
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.frames(), 3);
        assert_eq!(first.num_codebooks(), 2);
        assert_eq!(first.as_frame_major().len(), 6);
        assert!(
            first
                .as_frame_major()
                .iter()
                .all(|&token| token < config.vocab_size as u32)
        );
        assert_eq!(first.frame(0).unwrap().len(), 2);
        assert!(first.frame(3).is_err());
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
