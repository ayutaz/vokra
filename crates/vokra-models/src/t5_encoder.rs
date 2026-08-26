//! Shared native T5 encoder used by text-conditioned audio models.
//!
//! MusicGen, AudioGen, JASCO and AudioLDM2 all ship a frozen T5 text tower;
//! MT3 uses the same relative-position attention family for audio-token
//! transcription.  This module supplies one strict GGUF tensor walk and one
//! CPU/Metal forward instead of leaving five model families with independent
//! loud-partial placeholders.
//!
//! The implementation follows the Apache-2.0 Transformers `T5EncoderModel`:
//! gamma-only pre-norm, unscaled dot-product attention, one learned
//! bidirectional relative-position table shared across layers, ReLU or
//! gated-GELU feed-forward blocks, residual connections and a final gamma-only
//! norm. QKV/FFN projections, attention reductions, softmax, RMSNorm and the
//! selected activation all run through [`Compute`]. Embedding lookup, head
//! layout, gated-activation multiplication and relative-bias gathering are
//! deterministic host glue. Selecting Metal requires coverage for the full
//! learned-op set up front; there is no per-op CPU fallback.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};
use vokra_ops::t5_relative_position::{T5RelativePositionAttrs, t5_relative_attention_bias};

use crate::compute::{Compute, HotOp};

/// Learned operations required by a complete T5-base ReLU encoder forward.
///
/// Kept as the original public constant for AudioCraft callers. Configured
/// encoders select the exact ReLU or gated-GELU set at forward entry.
pub const T5_ENCODER_HOT_OPS: &[HotOp] =
    &[HotOp::Gemm, HotOp::Softmax, HotOp::RmsNorm, HotOp::Relu];

/// Learned operations required by a FLAN-T5 gated-GELU encoder forward.
pub const T5_GATED_GELU_ENCODER_HOT_OPS: &[HotOp] =
    &[HotOp::Gemm, HotOp::Softmax, HotOp::RmsNorm, HotOp::GeluNew];

/// T5 feed-forward projection family. The choice controls both the required
/// GGUF tensor names and the activation dispatched by [`Compute`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum T5FeedForwardKind {
    /// `wi(x)` followed by ReLU, used by canonical T5-base.
    Relu,
    /// `gelu_new(wi_0(x)) * wi_1(x)`, used by FLAN-T5-large in Parler-TTS.
    GatedGelu,
}

/// Canonical T5-base topology used by AudioCraft text conditioners.
pub const T5_BASE_CONFIG: T5EncoderConfig = T5EncoderConfig {
    vocab_size: 32_128,
    d_model: 768,
    d_kv: 64,
    d_ff: 3_072,
    num_layers: 12,
    num_heads: 12,
    relative_attention_num_buckets: 32,
    relative_attention_max_distance: 128,
    layer_norm_epsilon: 1.0e-6,
    feed_forward_kind: T5FeedForwardKind::Relu,
};

/// FLAN-T5-large topology embedded in both public Parler-TTS Mini GGUFs.
pub const FLAN_T5_LARGE_CONFIG: T5EncoderConfig = T5EncoderConfig {
    vocab_size: 32_128,
    d_model: 1_024,
    d_kv: 64,
    d_ff: 2_816,
    num_layers: 24,
    num_heads: 16,
    relative_attention_num_buckets: 32,
    relative_attention_max_distance: 128,
    layer_norm_epsilon: 1.0e-6,
    feed_forward_kind: T5FeedForwardKind::GatedGelu,
};

/// Explicit T5 encoder geometry. No topology axis is inferred from a partial
/// tensor set during execution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct T5EncoderConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub layer_norm_epsilon: f32,
    pub feed_forward_kind: T5FeedForwardKind,
}

impl T5EncoderConfig {
    /// Validate every shape relationship before weights are decoded.
    pub fn validate(self) -> Result<()> {
        if self.vocab_size == 0
            || self.d_model == 0
            || self.d_kv == 0
            || self.d_ff == 0
            || self.num_layers == 0
            || self.num_heads == 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "T5 encoder axes must be non-zero, got {self:?}"
            )));
        }
        self.num_heads
            .checked_mul(self.d_kv)
            .ok_or_else(|| VokraError::InvalidArgument("T5 num_heads*d_kv overflow".to_owned()))?;
        self.vocab_size.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument("T5 vocab_size*d_model overflow".to_owned())
        })?;
        self.d_model.checked_mul(self.inner_dim()).ok_or_else(|| {
            VokraError::InvalidArgument("T5 d_model*inner_dim overflow".to_owned())
        })?;
        self.d_model
            .checked_mul(self.d_ff)
            .ok_or_else(|| VokraError::InvalidArgument("T5 d_model*d_ff overflow".to_owned()))?;
        self.relative_attention_num_buckets
            .checked_mul(self.num_heads)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "T5 relative_attention_num_buckets*num_heads overflow".to_owned(),
                )
            })?;
        if !self.layer_norm_epsilon.is_finite() || self.layer_norm_epsilon <= 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "T5 layer_norm_epsilon must be finite and positive, got {}",
                self.layer_norm_epsilon
            )));
        }
        self.relative_attrs().validate()
    }

    fn inner_dim(self) -> usize {
        self.num_heads * self.d_kv
    }

    fn relative_attrs(self) -> T5RelativePositionAttrs {
        T5RelativePositionAttrs {
            num_buckets: self.relative_attention_num_buckets,
            max_distance: self.relative_attention_max_distance,
            bidirectional: true,
        }
    }
}

#[derive(Debug)]
struct T5AttentionWeights {
    /// Transposed at bind time from upstream `[out, in]` to GEMM `[in, out]`.
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
}

#[derive(Debug)]
enum T5FfnWeights {
    Relu {
        wi: Vec<f32>,
        wo: Vec<f32>,
    },
    GatedGelu {
        wi_0: Vec<f32>,
        wi_1: Vec<f32>,
        wo: Vec<f32>,
    },
}

#[derive(Debug)]
struct T5BlockWeights {
    self_attn_layer_norm: Vec<f32>,
    self_attn: T5AttentionWeights,
    ffn_layer_norm: Vec<f32>,
    ffn: T5FfnWeights,
}

/// Fully decoded T5 encoder weights. Composite GGUFs may carry other model
/// tensors; this binder requires every tensor under the requested T5 prefix
/// while leaving sibling tensors to their owning binder.
#[derive(Debug)]
pub struct T5EncoderWeights {
    shared_embedding: Vec<f32>,
    relative_attention_bias: Vec<f32>,
    blocks: Vec<T5BlockWeights>,
    final_layer_norm: Vec<f32>,
}

impl T5EncoderWeights {
    /// Bind a Transformers-style T5 encoder under `prefix` (for example
    /// `text_encoder` in MusicGen-family GGUFs).
    pub fn from_gguf(file: &GgufFile, prefix: &str, config: T5EncoderConfig) -> Result<Self> {
        config.validate()?;
        let prefix = normalize_prefix(prefix)?;
        let d = config.d_model;
        let inner = config.inner_dim();

        let shared_embedding = tensor(
            file,
            &format!("{prefix}.shared.weight"),
            &[config.vocab_size, d],
        )?;
        let relative_attention_bias = tensor(
            file,
            &format!(
                "{prefix}.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
            ),
            &[config.relative_attention_num_buckets, config.num_heads],
        )?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let base = format!("{prefix}.encoder.block.{layer}.layer");
            let attention = format!("{base}.0.SelfAttention");
            let ffn = format!("{base}.1.DenseReluDense");
            blocks.push(T5BlockWeights {
                self_attn_layer_norm: tensor(file, &format!("{base}.0.layer_norm.weight"), &[d])?,
                self_attn: T5AttentionWeights {
                    q: tensor_transposed(file, &format!("{attention}.q.weight"), inner, d)?,
                    k: tensor_transposed(file, &format!("{attention}.k.weight"), inner, d)?,
                    v: tensor_transposed(file, &format!("{attention}.v.weight"), inner, d)?,
                    o: tensor_transposed(file, &format!("{attention}.o.weight"), d, inner)?,
                },
                ffn_layer_norm: tensor(file, &format!("{base}.1.layer_norm.weight"), &[d])?,
                ffn: match config.feed_forward_kind {
                    T5FeedForwardKind::Relu => T5FfnWeights::Relu {
                        wi: tensor_transposed(file, &format!("{ffn}.wi.weight"), config.d_ff, d)?,
                        wo: tensor_transposed(file, &format!("{ffn}.wo.weight"), d, config.d_ff)?,
                    },
                    T5FeedForwardKind::GatedGelu => T5FfnWeights::GatedGelu {
                        wi_0: tensor_transposed(
                            file,
                            &format!("{ffn}.wi_0.weight"),
                            config.d_ff,
                            d,
                        )?,
                        wi_1: tensor_transposed(
                            file,
                            &format!("{ffn}.wi_1.weight"),
                            config.d_ff,
                            d,
                        )?,
                        wo: tensor_transposed(file, &format!("{ffn}.wo.weight"), d, config.d_ff)?,
                    },
                },
            });
        }
        let final_layer_norm = tensor(
            file,
            &format!("{prefix}.encoder.final_layer_norm.weight"),
            &[d],
        )?;

        Ok(Self {
            shared_embedding,
            relative_attention_bias,
            blocks,
            final_layer_norm,
        })
    }
}

/// Native T5 encoder with explicit CPU/Metal selection.
#[derive(Debug)]
pub struct T5Encoder {
    config: T5EncoderConfig,
    weights: T5EncoderWeights,
    backend: BackendKind,
}

impl T5Encoder {
    /// Bind canonical T5-base weights under `prefix` and select CPU.
    pub fn t5_base_from_gguf(file: &GgufFile, prefix: &str) -> Result<Self> {
        Self::from_gguf(file, prefix, T5_BASE_CONFIG)
    }

    /// Bind a fully specified T5 encoder and select CPU.
    pub fn from_gguf(file: &GgufFile, prefix: &str, config: T5EncoderConfig) -> Result<Self> {
        let weights = T5EncoderWeights::from_gguf(file, prefix, config)?;
        Ok(Self {
            config,
            weights,
            backend: BackendKind::Cpu,
        })
    }

    /// Construct from already validated weights (used by focused parity and
    /// composite model binders).
    pub fn new(config: T5EncoderConfig, weights: T5EncoderWeights) -> Result<Self> {
        config.validate()?;
        if weights.blocks.len() != config.num_layers {
            return Err(VokraError::ModelLoad(format!(
                "T5 weights carry {} blocks, expected {}",
                weights.blocks.len(),
                config.num_layers
            )));
        }
        Ok(Self {
            config,
            weights,
            backend: BackendKind::Cpu,
        })
    }

    /// Select the model backend. Availability and complete op coverage are
    /// checked at forward entry, before any token is processed.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    #[must_use]
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    #[must_use]
    pub fn config(&self) -> T5EncoderConfig {
        self.config
    }

    /// Encode token ids with an optional key mask. Returned layout is
    /// `[sequence, d_model]` row-major.
    pub fn encode_tokens(
        &self,
        token_ids: &[u32],
        attention_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(VokraError::InvalidArgument(
                "T5 encode_tokens requires at least one token".to_owned(),
            ));
        }
        if let Some(mask) = attention_mask {
            if mask.len() != token_ids.len() {
                return Err(VokraError::InvalidArgument(format!(
                    "T5 attention mask length {} != token length {}",
                    mask.len(),
                    token_ids.len()
                )));
            }
        }
        for (position, &token) in token_ids.iter().enumerate() {
            if token as usize >= self.config.vocab_size {
                return Err(VokraError::InvalidArgument(format!(
                    "T5 token_ids[{position}]={token} is outside vocab_size {}",
                    self.config.vocab_size
                )));
            }
        }

        let hot_ops = match self.config.feed_forward_kind {
            T5FeedForwardKind::Relu => T5_ENCODER_HOT_OPS,
            T5FeedForwardKind::GatedGelu => T5_GATED_GELU_ENCODER_HOT_OPS,
        };
        let compute = Compute::for_backend(self.backend, hot_ops)?;
        let sequence = token_ids.len();
        let d = self.config.d_model;
        let hidden_len = checked_product(sequence, d, "sequence*d_model")?;
        let mut hidden = Vec::with_capacity(hidden_len);
        for &token in token_ids {
            let start = token as usize * d;
            hidden.extend_from_slice(&self.weights.shared_embedding[start..start + d]);
        }
        let position_bias = t5_relative_attention_bias(
            &self.weights.relative_attention_bias,
            self.config.num_heads,
            sequence,
            sequence,
            self.config.relative_attrs(),
        )?;

        for block in &self.weights.blocks {
            self.forward_block(
                &compute,
                block,
                &position_bias,
                attention_mask,
                sequence,
                &mut hidden,
            )?;
        }

        let mut output = vec![0.0; hidden.len()];
        compute.rms_norm_f32(
            &hidden,
            &mut output,
            sequence,
            d,
            &self.weights.final_layer_norm,
            self.config.layer_norm_epsilon,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_block(
        &self,
        compute: &Compute,
        block: &T5BlockWeights,
        position_bias: &[f32],
        attention_mask: Option<&[bool]>,
        sequence: usize,
        hidden: &mut [f32],
    ) -> Result<()> {
        let d = self.config.d_model;
        let inner = self.config.inner_dim();
        let mut normalized = vec![0.0; hidden.len()];
        compute.rms_norm_f32(
            hidden,
            &mut normalized,
            sequence,
            d,
            &block.self_attn_layer_norm,
            self.config.layer_norm_epsilon,
        )?;

        let projected_len = checked_product(sequence, inner, "sequence*inner_dim")?;
        let mut q = vec![0.0; projected_len];
        let mut k = vec![0.0; projected_len];
        let mut v = vec![0.0; projected_len];
        compute.gemm_f32(
            sequence,
            inner,
            d,
            &normalized,
            &block.self_attn.q,
            None,
            &mut q,
        )?;
        compute.gemm_f32(
            sequence,
            inner,
            d,
            &normalized,
            &block.self_attn.k,
            None,
            &mut k,
        )?;
        compute.gemm_f32(
            sequence,
            inner,
            d,
            &normalized,
            &block.self_attn.v,
            None,
            &mut v,
        )?;

        let mut context = vec![0.0; projected_len];
        let head_dim = self.config.d_kv;
        let score_len = sequence.checked_mul(sequence).ok_or_else(|| {
            VokraError::InvalidArgument("T5 attention score length overflow".to_owned())
        })?;
        for head in 0..self.config.num_heads {
            let mut q_head = vec![0.0; sequence * head_dim];
            let mut k_head_t = vec![0.0; head_dim * sequence];
            let mut v_head = vec![0.0; sequence * head_dim];
            for row in 0..sequence {
                let source = row * inner + head * head_dim;
                let target = row * head_dim;
                q_head[target..target + head_dim].copy_from_slice(&q[source..source + head_dim]);
                v_head[target..target + head_dim].copy_from_slice(&v[source..source + head_dim]);
                for channel in 0..head_dim {
                    k_head_t[channel * sequence + row] = k[source + channel];
                }
            }

            let mut scores = vec![0.0; score_len];
            compute.gemm_f32(
                sequence,
                sequence,
                head_dim,
                &q_head,
                &k_head_t,
                None,
                &mut scores,
            )?;
            let bias = &position_bias[head * score_len..(head + 1) * score_len];
            for query in 0..sequence {
                for key in 0..sequence {
                    let index = query * sequence + key;
                    scores[index] += bias[index];
                    if attention_mask.is_some_and(|mask| !mask[key]) {
                        // Transformers expands an encoder mask with
                        // `(1 - mask) * torch.finfo(float32).min`, rather
                        // than negative infinity. Matching that finite
                        // sentinel also keeps the all-masked edge case
                        // deterministic instead of producing NaNs.
                        scores[index] += f32::MIN;
                    }
                }
            }
            let mut probabilities = vec![0.0; score_len];
            compute.softmax_f32(&scores, &mut probabilities, sequence, sequence)?;
            let mut head_context = vec![0.0; sequence * head_dim];
            compute.gemm_f32(
                sequence,
                head_dim,
                sequence,
                &probabilities,
                &v_head,
                None,
                &mut head_context,
            )?;
            for row in 0..sequence {
                let source = row * head_dim;
                let target = row * inner + head * head_dim;
                context[target..target + head_dim]
                    .copy_from_slice(&head_context[source..source + head_dim]);
            }
        }

        let mut attention_output = vec![0.0; hidden.len()];
        compute.gemm_f32(
            sequence,
            d,
            inner,
            &context,
            &block.self_attn.o,
            None,
            &mut attention_output,
        )?;
        add_assign(hidden, &attention_output)?;

        compute.rms_norm_f32(
            hidden,
            &mut normalized,
            sequence,
            d,
            &block.ffn_layer_norm,
            self.config.layer_norm_epsilon,
        )?;
        let ffn_len = checked_product(sequence, self.config.d_ff, "sequence*d_ff")?;
        let (activated, wo) = match &block.ffn {
            T5FfnWeights::Relu { wi, wo } => {
                let mut ffn_hidden = vec![0.0; ffn_len];
                compute.gemm_f32(
                    sequence,
                    self.config.d_ff,
                    d,
                    &normalized,
                    wi,
                    None,
                    &mut ffn_hidden,
                )?;
                let mut activated = vec![0.0; ffn_hidden.len()];
                compute.relu_f32(&ffn_hidden, &mut activated)?;
                (activated, wo)
            }
            T5FfnWeights::GatedGelu { wi_0, wi_1, wo } => {
                let mut preactivation = vec![0.0; ffn_len];
                compute.gemm_f32(
                    sequence,
                    self.config.d_ff,
                    d,
                    &normalized,
                    wi_0,
                    None,
                    &mut preactivation,
                )?;
                let mut activated = vec![0.0; ffn_len];
                compute.gelu_new_f32(&preactivation, &mut activated)?;
                let mut gate = vec![0.0; ffn_len];
                compute.gemm_f32(
                    sequence,
                    self.config.d_ff,
                    d,
                    &normalized,
                    wi_1,
                    None,
                    &mut gate,
                )?;
                for (activated, gate) in activated.iter_mut().zip(gate) {
                    *activated *= gate;
                }
                (activated, wo)
            }
        };
        let mut ffn_output = vec![0.0; hidden.len()];
        compute.gemm_f32(
            sequence,
            d,
            self.config.d_ff,
            &activated,
            wo,
            None,
            &mut ffn_output,
        )?;
        add_assign(hidden, &ffn_output)
    }
}

fn normalize_prefix(prefix: &str) -> Result<&str> {
    let normalized = prefix.trim_end_matches('.');
    if normalized.is_empty() || normalized != prefix {
        return Err(VokraError::InvalidArgument(format!(
            "T5 tensor prefix must be non-empty and omit the trailing dot, got {prefix:?}"
        )));
    }
    Ok(normalized)
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("T5 encoder is missing required tensor `{name}`"))
    })?;
    let actual: Vec<usize> = info
        .dimensions
        .iter()
        .map(|&axis| {
            usize::try_from(axis).map_err(|_| {
                VokraError::ModelLoad(format!(
                    "T5 tensor `{name}` axis {axis} does not fit this target's usize"
                ))
            })
        })
        .collect::<Result<_>>()?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "T5 tensor `{name}` shape {actual:?} != expected {expected:?}"
        )));
    }
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!(
            "T5 tensor `{name}` could not decode to f32: {error}"
        ))
    })
}

fn tensor_transposed(
    file: &GgufFile,
    name: &str,
    output_dim: usize,
    input_dim: usize,
) -> Result<Vec<f32>> {
    let source = tensor(file, name, &[output_dim, input_dim])?;
    let mut transposed = vec![0.0; source.len()];
    for output in 0..output_dim {
        for input in 0..input_dim {
            transposed[input * output_dim + output] = source[output * input_dim + input];
        }
    }
    Ok(transposed)
}

fn add_assign(destination: &mut [f32], source: &[f32]) -> Result<()> {
    if destination.len() != source.len() {
        return Err(VokraError::InvalidArgument(format!(
            "T5 residual length {} != source length {}",
            destination.len(),
            source.len()
        )));
    }
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += source;
    }
    Ok(())
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        VokraError::InvalidArgument(format!("T5 {label} length overflow: {left}*{right}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    const TINY: T5EncoderConfig = T5EncoderConfig {
        vocab_size: 8,
        d_model: 4,
        d_kv: 2,
        d_ff: 6,
        num_layers: 2,
        num_heads: 3,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 16,
        layer_norm_epsilon: 1.0e-6,
        feed_forward_kind: T5FeedForwardKind::Relu,
    };

    const GATED_TINY: T5EncoderConfig = T5EncoderConfig {
        feed_forward_kind: T5FeedForwardKind::GatedGelu,
        ..TINY
    };

    fn values(seed: usize, len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| (((index * 17 + seed * 13) % 29) as f32 - 14.0) / 37.0)
            .collect()
    }

    fn add_tensor(builder: &mut GgufBuilder, name: &str, shape: &[usize], seed: usize) {
        let len = shape.iter().product();
        let bytes = values(seed, len)
            .into_iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        builder
            .add_tensor(
                name,
                GgmlType::F32,
                shape.iter().map(|&axis| axis as u64).collect(),
                bytes,
            )
            .expect("tensor");
    }

    fn tiny_gguf_for(config: T5EncoderConfig, gate_seed: usize) -> GgufFile {
        let mut builder = GgufBuilder::new();
        add_tensor(&mut builder, "text.shared.weight", &[8, 4], 1);
        add_tensor(
            &mut builder,
            "text.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            &[8, 3],
            2,
        );
        for layer in 0..2 {
            let base = format!("text.encoder.block.{layer}.layer");
            add_tensor(
                &mut builder,
                &format!("{base}.0.layer_norm.weight"),
                &[4],
                10 + layer,
            );
            for (offset, projection) in ["q", "k", "v"].into_iter().enumerate() {
                add_tensor(
                    &mut builder,
                    &format!("{base}.0.SelfAttention.{projection}.weight"),
                    &[6, 4],
                    20 + layer * 10 + offset,
                );
            }
            add_tensor(
                &mut builder,
                &format!("{base}.0.SelfAttention.o.weight"),
                &[4, 6],
                23 + layer * 10,
            );
            add_tensor(
                &mut builder,
                &format!("{base}.1.layer_norm.weight"),
                &[4],
                30 + layer,
            );
            match config.feed_forward_kind {
                T5FeedForwardKind::Relu => add_tensor(
                    &mut builder,
                    &format!("{base}.1.DenseReluDense.wi.weight"),
                    &[6, 4],
                    40 + layer,
                ),
                T5FeedForwardKind::GatedGelu => {
                    add_tensor(
                        &mut builder,
                        &format!("{base}.1.DenseReluDense.wi_0.weight"),
                        &[6, 4],
                        40 + layer,
                    );
                    add_tensor(
                        &mut builder,
                        &format!("{base}.1.DenseReluDense.wi_1.weight"),
                        &[6, 4],
                        gate_seed + layer,
                    );
                }
            }
            add_tensor(
                &mut builder,
                &format!("{base}.1.DenseReluDense.wo.weight"),
                &[4, 6],
                50 + layer,
            );
        }
        add_tensor(
            &mut builder,
            "text.encoder.final_layer_norm.weight",
            &[4],
            60,
        );
        GgufFile::parse(builder.to_bytes().expect("serialize")).expect("parse")
    }

    fn tiny_gguf() -> GgufFile {
        tiny_gguf_for(TINY, 70)
    }

    #[test]
    fn tiny_cpu_forward_is_finite_deterministic_and_mask_sensitive() {
        let file = tiny_gguf();
        let encoder = T5Encoder::from_gguf(&file, "text", TINY).expect("bind");
        let a = encoder
            .encode_tokens(&[2, 3, 4], Some(&[true, true, true]))
            .expect("forward");
        let b = encoder
            .encode_tokens(&[2, 3, 4], Some(&[true, true, true]))
            .expect("forward repeat");
        let masked = encoder
            .encode_tokens(&[2, 3, 4], Some(&[true, true, false]))
            .expect("masked forward");
        assert_eq!(a, b);
        assert_eq!(a.len(), 3 * TINY.d_model);
        assert!(a.iter().all(|value| value.is_finite()));
        assert_ne!(a, masked);
    }

    #[test]
    fn gated_gelu_binds_distinct_projections_and_uses_the_gate() {
        let encoder = T5Encoder::from_gguf(&tiny_gguf_for(GATED_TINY, 70), "text", GATED_TINY)
            .expect("bind gated encoder");
        let changed_gate = T5Encoder::from_gguf(&tiny_gguf_for(GATED_TINY, 71), "text", GATED_TINY)
            .expect("bind changed gate");
        let output = encoder
            .encode_tokens(&[2, 3, 4], Some(&[true, true, true]))
            .expect("gated forward");
        let changed = changed_gate
            .encode_tokens(&[2, 3, 4], Some(&[true, true, true]))
            .expect("changed gated forward");
        assert_eq!(output.len(), 3 * GATED_TINY.d_model);
        assert!(output.iter().all(|value| value.is_finite()));
        assert_ne!(output, changed);
    }

    #[test]
    fn feed_forward_tensor_family_is_fail_closed() {
        let relu_error = T5Encoder::from_gguf(&tiny_gguf(), "text", GATED_TINY).unwrap_err();
        assert!(relu_error.to_string().contains("wi_0.weight"));

        let gated_error =
            T5Encoder::from_gguf(&tiny_gguf_for(GATED_TINY, 70), "text", TINY).unwrap_err();
        assert!(gated_error.to_string().contains("wi.weight"));
    }

    #[test]
    fn bind_rejects_missing_or_wrong_shape_tensor() {
        let mut builder = GgufBuilder::new();
        add_tensor(&mut builder, "text.shared.weight", &[8, 5], 1);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let error = T5Encoder::from_gguf(&file, "text", TINY).unwrap_err();
        assert!(error.to_string().contains("shape"));
    }

    #[test]
    fn token_and_mask_contracts_fail_closed() {
        let encoder = T5Encoder::from_gguf(&tiny_gguf(), "text", TINY).unwrap();
        assert!(encoder.encode_tokens(&[], None).is_err());
        assert!(encoder.encode_tokens(&[8], None).is_err());
        assert!(encoder.encode_tokens(&[1, 2], Some(&[true])).is_err());
    }

    #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
    #[test]
    fn metal_selection_without_feature_is_explicit_error() {
        let encoder = T5Encoder::from_gguf(&tiny_gguf(), "text", TINY)
            .unwrap()
            .with_backend(BackendKind::Metal);
        assert!(matches!(
            encoder.encode_tokens(&[1], None),
            Err(VokraError::BackendUnavailable(_))
        ));
    }
}
