//! Strict Qwen2 decoder used by the authenticated VibeVoice composite.
//!
//! The VibeVoice decoder is a Qwen2-family language model, but it is not the
//! Qwen implementation used by another product.  This module keeps the
//! VibeVoice dimensions and tensor names explicit and runs every learned
//! operation through the selected [`Compute`] backend.  The surrounding
//! composite (text processor, diffusion head and both streaming tokenizers)
//! is still a separate contract.

use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{load_tensor, require_tensor_shape};

/// The complete learned-operation set for the Qwen2 decoder.
pub const QWEN2_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::RmsNorm,
    HotOp::Silu,
];

/// Qwen vocabulary id for the VibeVoice speech-start marker.
pub const SPEECH_START_TOKEN_ID: u32 = 151_652;
/// Qwen vocabulary id for the VibeVoice speech-end marker.
pub const SPEECH_END_TOKEN_ID: u32 = 151_653;
/// Qwen vocabulary id for a diffusion placeholder position.
pub const SPEECH_DIFFUSION_TOKEN_ID: u32 = 151_654;
/// Qwen vocabulary id used by the fast tokenizer for padding.
pub const FAST_PADDING_TOKEN_ID: u32 = 151_655;
/// Qwen vocabulary id for BOS and EOS in the fixed companion tokenizer.
pub const BOS_EOS_TOKEN_ID: u32 = 151_643;

/// Fixed Qwen2 axes authenticated from the VibeVoice-1.5B checkpoint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen2RuntimeConfig {
    /// Model hidden width.
    pub hidden_size: usize,
    /// Vocabulary size and tied embedding/output rows.
    pub vocab_size: usize,
    /// Number of decoder blocks.
    pub num_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of grouped key/value heads.
    pub num_key_value_heads: usize,
    /// Feed-forward intermediate width.
    pub intermediate_size: usize,
    /// Rotary embedding base.
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Maximum supported position.
    pub max_position_embeddings: usize,
}

impl Qwen2RuntimeConfig {
    /// The fixed VibeVoice 1.5B Qwen2 decoder configuration.
    #[must_use]
    pub const fn vibevoice_1_5b() -> Self {
        Self {
            hidden_size: 1_536,
            vocab_size: 151_936,
            num_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            intermediate_size: 8_960,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1.0e-6,
            max_position_embeddings: 65_536,
        }
    }

    fn validate(self) -> Result<()> {
        if self.hidden_size == 0
            || self.vocab_size == 0
            || self.num_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.intermediate_size == 0
            || self.max_position_embeddings == 0
            || self.num_attention_heads % self.num_key_value_heads != 0
            || self.hidden_size % self.num_attention_heads != 0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
        {
            return Err(VokraError::InvalidArgument(
                "vibevoice Qwen2 configuration violates fixed GQA/RoPE axes".to_owned(),
            ));
        }
        Ok(())
    }

    fn head_dim(self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn kv_width(self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }
}

#[derive(Debug, Clone)]
struct Linear {
    /// Row-major `[in_features, out_features]` matrix for `Compute::gemm_f32`.
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    fn apply(&self, compute: &Compute, input: &[f32], rows: usize) -> Result<Vec<f32>> {
        if rows == 0 || input.len() != rows * self.in_features {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice Qwen2 linear input shape mismatch: rows={rows}, input={}, expected {}",
                input.len(),
                rows * self.in_features
            )));
        }
        let mut output = vec![0.0; rows * self.out_features];
        compute.gemm_f32(
            rows,
            self.out_features,
            self.in_features,
            input,
            &self.weight,
            self.bias.as_deref(),
            &mut output,
        )?;
        finite("Qwen2 linear output", &output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    gate: Linear,
    up: Linear,
    down: Linear,
}

#[derive(Debug, Clone)]
struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl LayerCache {
    fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }
}

/// Tied Qwen2 weights bound to the fixed VibeVoice tensor layout.
#[derive(Debug)]
pub(crate) struct Qwen2Weights {
    config: Qwen2RuntimeConfig,
    embedding: Vec<f32>,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
}

impl Qwen2Weights {
    /// Loads every Qwen2 tensor from the authenticated GGUF.
    ///
    /// The caller must first apply [`crate::vibevoice::VibeVoiceCheckpoint::from_gguf`]; this
    /// method repeats all local shape checks but does not treat a count-only
    /// or synthetic GGUF as authenticated.
    pub(crate) fn from_gguf(file: &GgufFile) -> Result<Self> {
        let config = Qwen2RuntimeConfig::vibevoice_1_5b();
        config.validate()?;
        let embedding = load_raw(
            file,
            "model.language_model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
        let final_norm = load_raw(
            file,
            "model.language_model.norm.weight",
            &[config.hidden_size],
        )?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            let prefix = format!("model.language_model.layers.{index}");
            layers.push(Layer {
                q: load_linear(
                    file,
                    &format!("{prefix}.self_attn.q_proj"),
                    config.hidden_size,
                    config.hidden_size,
                    true,
                )?,
                k: load_linear(
                    file,
                    &format!("{prefix}.self_attn.k_proj"),
                    config.hidden_size,
                    config.kv_width(),
                    true,
                )?,
                v: load_linear(
                    file,
                    &format!("{prefix}.self_attn.v_proj"),
                    config.hidden_size,
                    config.kv_width(),
                    true,
                )?,
                o: load_linear(
                    file,
                    &format!("{prefix}.self_attn.o_proj"),
                    config.hidden_size,
                    config.hidden_size,
                    false,
                )?,
                input_norm: load_raw(
                    file,
                    &format!("{prefix}.input_layernorm.weight"),
                    &[config.hidden_size],
                )?,
                post_norm: load_raw(
                    file,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    &[config.hidden_size],
                )?,
                gate: load_linear(
                    file,
                    &format!("{prefix}.mlp.gate_proj"),
                    config.hidden_size,
                    config.intermediate_size,
                    false,
                )?,
                up: load_linear(
                    file,
                    &format!("{prefix}.mlp.up_proj"),
                    config.hidden_size,
                    config.intermediate_size,
                    false,
                )?,
                down: load_linear(
                    file,
                    &format!("{prefix}.mlp.down_proj"),
                    config.intermediate_size,
                    config.hidden_size,
                    false,
                )?,
            });
        }
        Ok(Self {
            config,
            embedding,
            layers,
            final_norm,
        })
    }
}

/// A selected-backend Qwen2 decoder with reusable per-layer KV cache.
#[derive(Debug, Clone)]
pub struct Qwen2Runtime {
    weights: Arc<Qwen2Weights>,
    backend: BackendKind,
    cache: Vec<LayerCache>,
    position: usize,
}

impl Qwen2Runtime {
    /// Binds Qwen2 weights and preflights the complete learned-op backend set.
    pub(crate) fn new(weights: Qwen2Weights, backend: BackendKind) -> Result<Self> {
        weights.config.validate()?;
        validate_weight_shapes(&weights)?;
        let _ = Compute::for_backend(backend, QWEN2_HOT_OPS)?;
        let cache = (0..weights.config.num_layers)
            .map(|_| LayerCache::new())
            .collect();
        Ok(Self {
            weights: Arc::new(weights),
            backend,
            cache,
            position: 0,
        })
    }

    /// Loads and binds the Qwen2 section of an authenticated VibeVoice GGUF.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        // Authentication is enforced here rather than delegated to callers.
        super::VibeVoiceCheckpoint::from_gguf(file)?;
        Self::new(Qwen2Weights::from_gguf(file)?, backend)
    }

    /// Returns the selected backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the fixed decoder configuration.
    #[must_use]
    pub fn config(&self) -> Qwen2RuntimeConfig {
        self.weights.config
    }

    /// Clears the KV cache and starts a new sequence at position zero.
    pub fn reset(&mut self) {
        for layer in &mut self.cache {
            layer.clear();
        }
        self.position = 0;
    }

    /// Forks an independent generation cache while sharing immutable model
    /// weights. Positive and negative CFG branches must not share KV state;
    /// this creates empty branch caches without copying the checkpoint.
    #[must_use]
    pub fn fork_empty_cache(&self) -> Self {
        let cache = (0..self.weights.config.num_layers)
            .map(|_| LayerCache::new())
            .collect();
        Self {
            weights: Arc::clone(&self.weights),
            backend: self.backend,
            cache,
            position: 0,
        }
    }

    /// Runs a complete causal prompt matrix and returns one hidden row per
    /// input row. The resulting KV cache can be continued with [`Self::step`]
    /// or [`Self::step_embedding`].
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vibevoice Qwen2 prefill requires at least one token".to_owned(),
            ));
        }
        if tokens.len() > self.config().max_position_embeddings {
            return Err(VokraError::InvalidArgument(
                "vibevoice Qwen2 prefill exceeds max_position_embeddings".to_owned(),
            ));
        }
        if tokens
            .iter()
            .any(|&token| token as usize >= self.config().vocab_size)
        {
            return Err(VokraError::InvalidArgument(
                "vibevoice Qwen2 prefill token is outside vocabulary".to_owned(),
            ));
        }
        let d = self.config().hidden_size;
        let mut embeddings = Vec::with_capacity(tokens.len() * d);
        for &token in tokens {
            let start = token as usize * d;
            embeddings.extend_from_slice(&self.weights.embedding[start..start + d]);
        }
        self.prefill_embeddings(&embeddings, tokens.len())
    }

    /// Runs one autoregressive token using the existing KV cache.
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        if usize::try_from(token).map_or(true, |id| id >= self.config().vocab_size) {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice Qwen2 token {token} is outside vocabulary"
            )));
        }
        let d = self.config().hidden_size;
        let start = token as usize * d;
        let embedding = self.weights.embedding[start..start + d].to_vec();
        self.step_embedding(&embedding)
    }

    /// Runs a full causal prompt whose rows are already mixed embeddings.
    ///
    /// This is required for VibeVoice audio rows: the acoustic and semantic
    /// connectors produce the next LM input embedding rather than a vocabulary
    /// token. `embeddings` is row-major `[rows, hidden_size]`.
    pub fn prefill_embeddings(&mut self, embeddings: &[f32], rows: usize) -> Result<Vec<f32>> {
        if rows == 0 || rows > self.config().max_position_embeddings {
            return Err(VokraError::InvalidArgument(
                "vibevoice Qwen2 embedding prefill row count is outside limits".to_owned(),
            ));
        }
        if embeddings.len() != rows * self.config().hidden_size {
            return Err(VokraError::InvalidArgument(
                "vibevoice Qwen2 embedding prefill shape mismatch".to_owned(),
            ));
        }
        finite("Qwen2 embedding prefill input", embeddings)?;
        // Evaluate against a fresh cache and commit only on success. Moving
        // the old cache avoids cloning potentially large KV tensors while
        // preserving a caller's previous valid generation state on errors.
        let new_cache = self.empty_cache();
        let previous_cache = std::mem::replace(&mut self.cache, new_cache);
        let previous_position = self.position;
        self.position = 0;
        match self.prefill_full(embeddings, rows) {
            Ok(output) => Ok(output),
            Err(error) => {
                self.cache = previous_cache;
                self.position = previous_position;
                Err(error)
            }
        }
    }

    /// Embeds a token prompt and replaces selected rows with caller-provided
    /// mixed embeddings, such as acoustic+semantic connector outputs.
    /// Replacement indices must be unique, in range, and have the fixed
    /// hidden width. The other rows remain vocabulary embeddings.
    pub fn prefill_mixed_embeddings(
        &mut self,
        tokens: &[u32],
        replacements: &[(usize, &[f32])],
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() || tokens.len() > self.config().max_position_embeddings {
            return Err(VokraError::InvalidArgument(
                "vibevoice Qwen2 mixed prefill token count is outside limits".to_owned(),
            ));
        }
        let d = self.config().hidden_size;
        let mut embeddings = Vec::with_capacity(tokens.len() * d);
        for &token in tokens {
            let id = usize::try_from(token).map_err(|_| {
                VokraError::InvalidArgument("vibevoice Qwen2 token id conversion failed".to_owned())
            })?;
            if id >= self.config().vocab_size {
                return Err(VokraError::InvalidArgument(
                    "vibevoice Qwen2 mixed prefill token is outside vocabulary".to_owned(),
                ));
            }
            let start = id * d;
            embeddings.extend_from_slice(&self.weights.embedding[start..start + d]);
        }
        for (replacement_index, replacement) in replacements {
            if *replacement_index >= tokens.len() || replacement.len() != d {
                return Err(VokraError::InvalidArgument(
                    "vibevoice Qwen2 mixed prefill replacement shape/index mismatch".to_owned(),
                ));
            }
            if replacements
                .iter()
                .filter(|(index, _)| index == replacement_index)
                .count()
                != 1
            {
                return Err(VokraError::InvalidArgument(
                    "vibevoice Qwen2 mixed prefill replacement indices must be unique".to_owned(),
                ));
            }
            finite("Qwen2 mixed prefill replacement", replacement)?;
            let start = replacement_index * d;
            embeddings[start..start + d].copy_from_slice(replacement);
        }
        self.prefill_embeddings(&embeddings, tokens.len())
    }

    /// Runs one mixed embedding row using the existing KV cache.
    pub fn step_embedding(&mut self, embedding: &[f32]) -> Result<Vec<f32>> {
        let previous_lengths: Vec<(usize, usize)> = self
            .cache
            .iter()
            .map(|layer| (layer.keys.len(), layer.values.len()))
            .collect();
        let previous_position = self.position;
        let result = self.step_embedding_inner(embedding);
        if result.is_err() {
            for (layer, (keys_len, values_len)) in self.cache.iter_mut().zip(previous_lengths) {
                layer.keys.truncate(keys_len);
                layer.values.truncate(values_len);
            }
            self.position = previous_position;
        }
        result
    }

    fn step_embedding_inner(&mut self, embedding: &[f32]) -> Result<Vec<f32>> {
        if embedding.len() != self.config().hidden_size {
            return Err(VokraError::InvalidArgument(
                "vibevoice Qwen2 embedding step shape mismatch".to_owned(),
            ));
        }
        finite("Qwen2 embedding step input", embedding)?;
        if self.position >= self.config().max_position_embeddings {
            return Err(VokraError::InvalidArgument(
                "vibevoice Qwen2 position exceeds max_position_embeddings".to_owned(),
            ));
        }
        let compute = Compute::for_backend(self.backend, QWEN2_HOT_OPS)?;
        let mut hidden = embedding.to_vec();
        finite("Qwen2 embedding", &hidden)?;
        for layer_index in 0..self.weights.layers.len() {
            let layer = &self.weights.layers[layer_index];
            let normed = rms(
                &compute,
                &hidden,
                &layer.input_norm,
                self.config().rms_norm_eps,
            )?;
            let mut q = layer.q.apply(&compute, &normed, 1)?;
            let mut k = layer.k.apply(&compute, &normed, 1)?;
            let v = layer.v.apply(&compute, &normed, 1)?;
            apply_rope(
                &mut q,
                self.position,
                self.config().rope_theta,
                self.config().head_dim(),
            )?;
            apply_rope(
                &mut k,
                self.position,
                self.config().rope_theta,
                self.config().head_dim(),
            )?;
            let config = self.config();
            let attention =
                Self::attend(&mut self.cache, config, &compute, layer_index, &q, &k, &v)?;
            let projected = layer.o.apply(&compute, &attention, 1)?;
            add_assign(&mut hidden, &projected)?;
            let normed = rms(
                &compute,
                &hidden,
                &layer.post_norm,
                self.config().rms_norm_eps,
            )?;
            let mut gate = layer.gate.apply(&compute, &normed, 1)?;
            let up = layer.up.apply(&compute, &normed, 1)?;
            let gate_input = gate.clone();
            compute.silu_f32(&gate_input, &mut gate)?;
            for (gate_value, up_value) in gate.iter_mut().zip(up) {
                *gate_value *= up_value;
            }
            let projected = layer.down.apply(&compute, &gate, 1)?;
            add_assign(&mut hidden, &projected)?;
        }
        self.position += 1;
        rms(
            &compute,
            &hidden,
            &self.weights.final_norm,
            self.config().rms_norm_eps,
        )
    }

    fn empty_cache(&self) -> Vec<LayerCache> {
        (0..self.weights.config.num_layers)
            .map(|_| LayerCache::new())
            .collect()
    }

    /// Projects one hidden row through the tied vocabulary embedding.
    pub fn logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        if hidden.len() != self.config().hidden_size {
            return Err(VokraError::InvalidArgument(
                "vibevoice Qwen2 logits hidden shape mismatch".to_owned(),
            ));
        }
        let compute = Compute::for_backend(self.backend, QWEN2_HOT_OPS)?;
        let mut output = vec![0.0; self.config().vocab_size];
        compute.gemv_f32(
            self.config().vocab_size,
            self.config().hidden_size,
            &self.weights.embedding,
            hidden,
            None,
            &mut output,
        )?;
        finite("Qwen2 logits", &output)?;
        Ok(output)
    }

    fn prefill_full(&mut self, embeddings: &[f32], rows: usize) -> Result<Vec<f32>> {
        let config = self.config();
        let compute = Compute::for_backend(self.backend, QWEN2_HOT_OPS)?;
        let d = config.hidden_size;
        let mut hidden = embeddings.to_vec();
        finite("Qwen2 prefill embedding", &hidden)?;
        for layer_index in 0..self.weights.layers.len() {
            let layer = &self.weights.layers[layer_index];
            let normed = rms_rows(
                &compute,
                &hidden,
                &layer.input_norm,
                rows,
                d,
                config.rms_norm_eps,
            )?;
            let mut q = layer.q.apply(&compute, &normed, rows)?;
            let mut k = layer.k.apply(&compute, &normed, rows)?;
            let v = layer.v.apply(&compute, &normed, rows)?;
            for position in 0..rows {
                apply_rope(
                    &mut q[position * d..(position + 1) * d],
                    position,
                    config.rope_theta,
                    config.head_dim(),
                )?;
                apply_rope(
                    &mut k[position * config.kv_width()..(position + 1) * config.kv_width()],
                    position,
                    config.rope_theta,
                    config.head_dim(),
                )?;
            }
            self.cache[layer_index].keys = k.clone();
            self.cache[layer_index].values = v.clone();
            let mut attended = vec![0.0; rows * d];
            let scale = (config.head_dim() as f32).sqrt().recip();
            for position in 0..rows {
                let context = position + 1;
                for head in 0..config.num_attention_heads {
                    let kv_head = head / (config.num_attention_heads / config.num_key_value_heads);
                    let q_start = position * d + head * config.head_dim();
                    let mut key_matrix = vec![0.0; config.head_dim() * context];
                    for prior in 0..context {
                        for component in 0..config.head_dim() {
                            key_matrix[component * context + prior] = k[prior * config.kv_width()
                                + kv_head * config.head_dim()
                                + component];
                        }
                    }
                    let mut scores = vec![0.0; context];
                    compute.gemm_f32(
                        1,
                        context,
                        config.head_dim(),
                        &q[q_start..q_start + config.head_dim()],
                        &key_matrix,
                        None,
                        &mut scores,
                    )?;
                    for score in &mut scores {
                        *score *= scale;
                    }
                    let mut probabilities = vec![0.0; context];
                    compute.softmax_f32(&scores, &mut probabilities, 1, context)?;
                    let mut value_matrix = vec![0.0; context * config.head_dim()];
                    for prior in 0..context {
                        value_matrix[prior * config.head_dim()..(prior + 1) * config.head_dim()]
                            .copy_from_slice(
                                &v[prior * config.kv_width() + kv_head * config.head_dim()
                                    ..prior * config.kv_width()
                                        + (kv_head + 1) * config.head_dim()],
                            );
                    }
                    let mut head_output = vec![0.0; config.head_dim()];
                    compute.gemm_f32(
                        1,
                        config.head_dim(),
                        context,
                        &probabilities,
                        &value_matrix,
                        None,
                        &mut head_output,
                    )?;
                    attended[q_start..q_start + config.head_dim()].copy_from_slice(&head_output);
                }
            }
            let projected = layer.o.apply(&compute, &attended, rows)?;
            add_assign(&mut hidden, &projected)?;
            let normed = rms_rows(
                &compute,
                &hidden,
                &layer.post_norm,
                rows,
                d,
                config.rms_norm_eps,
            )?;
            let mut gate = layer.gate.apply(&compute, &normed, rows)?;
            let up = layer.up.apply(&compute, &normed, rows)?;
            let gate_input = gate.clone();
            compute.silu_f32(&gate_input, &mut gate)?;
            for (gate_value, up_value) in gate.iter_mut().zip(up) {
                *gate_value *= up_value;
            }
            let projected = layer.down.apply(&compute, &gate, rows)?;
            add_assign(&mut hidden, &projected)?;
        }
        self.position = rows;
        rms_rows(
            &compute,
            &hidden,
            &self.weights.final_norm,
            rows,
            d,
            config.rms_norm_eps,
        )
    }

    fn attend(
        cache: &mut [LayerCache],
        config: Qwen2RuntimeConfig,
        compute: &Compute,
        layer_index: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> Result<Vec<f32>> {
        let head_dim = config.head_dim();
        let kv_width = config.kv_width();
        let layer_cache = &mut cache[layer_index];
        layer_cache.keys.extend_from_slice(k);
        layer_cache.values.extend_from_slice(v);
        let context = layer_cache.keys.len() / kv_width;
        let mut output = vec![0.0; config.hidden_size];
        let scale = (head_dim as f32).sqrt().recip();
        for head in 0..config.num_attention_heads {
            let kv_head = head / (config.num_attention_heads / config.num_key_value_heads);
            let q_start = head * head_dim;
            let mut key_matrix = vec![0.0; head_dim * context];
            for position in 0..context {
                for component in 0..head_dim {
                    key_matrix[component * context + position] =
                        layer_cache.keys[position * kv_width + kv_head * head_dim + component];
                }
            }
            let mut scores = vec![0.0; context];
            compute.gemm_f32(
                1,
                context,
                head_dim,
                &q[q_start..q_start + head_dim],
                &key_matrix,
                None,
                &mut scores,
            )?;
            for score in &mut scores {
                *score *= scale;
            }
            let mut probabilities = vec![0.0; context];
            compute.softmax_f32(&scores, &mut probabilities, 1, context)?;
            let mut value_matrix = vec![0.0; context * head_dim];
            for position in 0..context {
                value_matrix[position * head_dim..(position + 1) * head_dim].copy_from_slice(
                    &layer_cache.values[position * kv_width + kv_head * head_dim
                        ..position * kv_width + (kv_head + 1) * head_dim],
                );
            }
            let mut head_output = vec![0.0; head_dim];
            compute.gemm_f32(
                1,
                head_dim,
                context,
                &probabilities,
                &value_matrix,
                None,
                &mut head_output,
            )?;
            output[q_start..q_start + head_dim].copy_from_slice(&head_output);
        }
        finite("Qwen2 attention", &output)?;
        Ok(output)
    }
}

fn load_raw(file: &GgufFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    require_tensor_shape(file, "vibevoice Qwen2", name, shape)?;
    load_tensor(file, "vibevoice Qwen2", name, shape)
}

fn load_linear(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<Linear> {
    let raw = load_raw(file, &format!("{prefix}.weight"), &[output, input])?;
    let mut weight = vec![0.0; input * output];
    for row in 0..output {
        for col in 0..input {
            weight[col * output + row] = raw[row * input + col];
        }
    }
    let bias = if bias {
        Some(load_raw(file, &format!("{prefix}.bias"), &[output])?)
    } else {
        None
    };
    Ok(Linear {
        weight,
        bias,
        in_features: input,
        out_features: output,
    })
}

fn validate_weight_shapes(weights: &Qwen2Weights) -> Result<()> {
    let config = weights.config;
    if weights.embedding.len() != config.vocab_size * config.hidden_size
        || weights.final_norm.len() != config.hidden_size
        || weights.layers.len() != config.num_layers
    {
        return Err(VokraError::ModelLoad(
            "vibevoice Qwen2 bound embedding/layer shape mismatch".to_owned(),
        ));
    }
    for layer in &weights.layers {
        require_linear(&layer.q, config.hidden_size, config.hidden_size, true)?;
        require_linear(&layer.k, config.hidden_size, config.kv_width(), true)?;
        require_linear(&layer.v, config.hidden_size, config.kv_width(), true)?;
        require_linear(&layer.o, config.hidden_size, config.hidden_size, false)?;
        require_linear(
            &layer.gate,
            config.hidden_size,
            config.intermediate_size,
            false,
        )?;
        require_linear(
            &layer.up,
            config.hidden_size,
            config.intermediate_size,
            false,
        )?;
        require_linear(
            &layer.down,
            config.intermediate_size,
            config.hidden_size,
            false,
        )?;
        if layer.input_norm.len() != config.hidden_size
            || layer.post_norm.len() != config.hidden_size
        {
            return Err(VokraError::ModelLoad(
                "vibevoice Qwen2 layer norm shape mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

fn require_linear(linear: &Linear, input: usize, output: usize, bias: bool) -> Result<()> {
    if linear.in_features != input
        || linear.out_features != output
        || linear.weight.len() != input * output
        || linear
            .bias
            .as_ref()
            .is_some_and(|value| value.len() != output)
        || (bias != linear.bias.is_some())
    {
        return Err(VokraError::ModelLoad(
            "vibevoice Qwen2 linear shape or bias contract mismatch".to_owned(),
        ));
    }
    Ok(())
}

fn rms(compute: &Compute, input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    let mut output = vec![0.0; input.len()];
    compute.rms_norm_f32(input, &mut output, 1, input.len(), weight, eps)?;
    finite("Qwen2 RMSNorm", &output)?;
    Ok(output)
}

fn rms_rows(
    compute: &Compute,
    input: &[f32],
    weight: &[f32],
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if rows == 0 || input.len() != rows * width {
        return Err(VokraError::InvalidArgument(
            "vibevoice Qwen2 row norm shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0; input.len()];
    compute.rms_norm_f32(input, &mut output, rows, width, weight, eps)?;
    finite("Qwen2 RMSNorm rows", &output)?;
    Ok(output)
}

fn add_assign(left: &mut [f32], right: &[f32]) -> Result<()> {
    if left.len() != right.len() {
        return Err(VokraError::InvalidArgument(
            "vibevoice Qwen2 residual shape mismatch".to_owned(),
        ));
    }
    for (left, right) in left.iter_mut().zip(right) {
        *left += *right;
    }
    finite("Qwen2 residual", left)
}

fn finite(label: &str, values: &[f32]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(format!(
            "{label} contains non-finite values"
        )));
    }
    Ok(())
}

fn apply_rope(values: &mut [f32], position: usize, theta: f32, head_dim: usize) -> Result<()> {
    if head_dim % 2 != 0 || values.len() % head_dim != 0 {
        return Err(VokraError::InvalidArgument(
            "vibevoice Qwen2 RoPE shape mismatch".to_owned(),
        ));
    }
    let heads = values.len() / head_dim;
    for head in 0..heads {
        let row = &mut values[head * head_dim..(head + 1) * head_dim];
        let half = head_dim / 2;
        for pair in 0..half {
            let exponent = (2 * pair) as f32 / head_dim as f32;
            let angle = position as f32 / theta.powf(exponent);
            let (sin, cos) = angle.sin_cos();
            let first = row[pair];
            let second = row[half + pair];
            // Transformers' rotate_half splits the head into two contiguous
            // halves.  This is intentionally not the interleaved convention
            // used by Zonos and several audio tokenizers.
            row[pair] = first * cos - second * sin;
            row[half + pair] = second * cos + first * sin;
        }
    }
    finite("Qwen2 RoPE", values)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn fixed_qwen2_axes_and_gqa_ratio() {
        let config = Qwen2RuntimeConfig::vibevoice_1_5b();
        assert_eq!(config.hidden_size, 1_536);
        assert_eq!(config.num_layers, 28);
        assert_eq!(config.vocab_size, 151_936);
        assert_eq!(config.num_attention_heads / config.num_key_value_heads, 6);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.kv_width(), 256);
    }

    #[test]
    fn gqa_maps_six_query_heads_to_each_kv_head() {
        let config = Qwen2RuntimeConfig {
            hidden_size: 512,
            vocab_size: 10,
            num_layers: 1,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            intermediate_size: 16,
            rope_theta: 1.0e6,
            rms_norm_eps: 1.0e-6,
            max_position_embeddings: 8,
        };
        let group = config.num_attention_heads / config.num_key_value_heads;
        let mapped: Vec<usize> = (0..config.num_attention_heads)
            .map(|head| head / group)
            .collect();
        assert_eq!(mapped, [0, 0, 0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn rope_zero_position_is_identity_and_changes_later_position() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let original = values.clone();
        apply_rope(&mut values, 0, 1.0e6, 4).unwrap();
        assert_eq!(values, original);
        apply_rope(&mut values, 1, 1.0e6, 4).unwrap();
        assert_ne!(values, original);
    }

    #[test]
    fn rope_uses_qwen_split_half_rotate_oracle() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut values, 1, 100.0, 4).unwrap();
        let (sin0, cos0) = 1.0_f32.sin_cos();
        let (sin1, cos1) = 0.1_f32.sin_cos();
        let expected = [
            1.0 * cos0 - 3.0 * sin0,
            2.0 * cos1 - 4.0 * sin1,
            3.0 * cos0 + 1.0 * sin0,
            4.0 * cos1 + 2.0 * sin1,
        ];
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn linear_transpose_layout_matches_row_major_gemm() {
        let raw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [out=2,in=3]
        let mut transposed = vec![0.0; 6];
        for row in 0..2 {
            for col in 0..3 {
                transposed[col * 2 + row] = raw[row * 3 + col];
            }
        }
        assert_eq!(transposed, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn prefill_and_incremental_step_share_the_same_kv_path() {
        let mut prefill = fixture_runtime();
        let input = [0.1_f32, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1];
        let all = prefill.prefill_embeddings(&input, 2).unwrap();
        let expected = reference_full(&prefill, &input, 2);
        for (actual, expected) in all.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
        let mut incremental = fixture_runtime();
        let first = incremental.step_embedding(&input[..4]).unwrap();
        let second = incremental.step_embedding(&input[4..]).unwrap();
        for (actual, expected) in all[..4].iter().zip(first) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        for (actual, expected) in all[4..].iter().zip(second) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn mixed_prefill_replaces_selected_audio_rows_only() {
        let mut mixed = fixture_runtime();
        let replacement = [0.8_f32, -0.7, 0.6, -0.5];
        let mixed_output = mixed
            .prefill_mixed_embeddings(&[0, 1], &[(1, &replacement)])
            .unwrap();

        let mut direct = fixture_runtime();
        let direct_input = [0.1_f32, 0.2, 0.3, 0.4, 0.8, -0.7, 0.6, -0.5];
        let direct_output = direct.prefill_embeddings(&direct_input, 2).unwrap();
        assert_eq!(mixed_output.len(), direct_output.len());
        for (actual, expected) in mixed_output.iter().zip(direct_output) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        let mut invalid = fixture_runtime();
        assert!(
            invalid
                .prefill_mixed_embeddings(&[0, 1], &[(1, &replacement), (1, &replacement)])
                .is_err()
        );
    }

    #[test]
    fn fork_empty_cache_shares_weights_but_not_generation_state() {
        let runtime = fixture_runtime();
        let fork = runtime.fork_empty_cache();
        assert_eq!(Arc::strong_count(&runtime.weights), 2);
        assert_eq!(runtime.position, 0);
        assert_eq!(fork.position, 0);
        assert!(fork.cache.iter().all(|layer| layer.keys.is_empty()));
    }

    #[test]
    fn prefill_error_restores_previous_cache_without_cloning_it() {
        let mut runtime = fixture_runtime();
        runtime.step_embedding(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        let previous_position = runtime.position;
        let previous_lengths: Vec<(usize, usize)> = runtime
            .cache
            .iter()
            .map(|layer| (layer.keys.len(), layer.values.len()))
            .collect();
        let weights = Arc::get_mut(&mut runtime.weights).unwrap();
        weights.layers[0].o.weight[0] = f32::NAN;
        assert!(
            runtime
                .prefill_embeddings(&[0.1, 0.2, 0.3, 0.4], 1)
                .is_err()
        );
        assert_eq!(runtime.position, previous_position);
        let lengths: Vec<(usize, usize)> = runtime
            .cache
            .iter()
            .map(|layer| (layer.keys.len(), layer.values.len()))
            .collect();
        assert_eq!(lengths, previous_lengths);
    }

    #[test]
    fn step_error_truncates_partial_attend_appends() {
        let mut runtime = fixture_runtime();
        let weights = Arc::get_mut(&mut runtime.weights).unwrap();
        weights.layers[0].o.weight[0] = f32::NAN;
        assert!(runtime.step_embedding(&[0.1, 0.2, 0.3, 0.4]).is_err());
        assert_eq!(runtime.position, 0);
        assert!(
            runtime
                .cache
                .iter()
                .all(|layer| { layer.keys.is_empty() && layer.values.is_empty() })
        );
    }

    fn fixture_runtime() -> Qwen2Runtime {
        let config = Qwen2RuntimeConfig {
            hidden_size: 4,
            vocab_size: 8,
            num_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            intermediate_size: 8,
            rope_theta: 1.0e6,
            rms_norm_eps: 1.0e-6,
            max_position_embeddings: 16,
        };
        let layer = Layer {
            q: identity_linear(4, 4, true),
            k: identity_linear(4, 2, true),
            v: identity_linear(4, 2, true),
            o: identity_linear(4, 4, false),
            input_norm: vec![1.0; 4],
            post_norm: vec![1.0; 4],
            gate: identity_linear(4, 8, false),
            up: identity_linear(4, 8, false),
            down: identity_linear(8, 4, false),
        };
        let embedding = (0..config.vocab_size * config.hidden_size)
            .map(|index| (index % config.hidden_size) as f32 * 0.1 + 0.1)
            .collect();
        Qwen2Runtime::new(
            Qwen2Weights {
                config,
                embedding,
                layers: vec![layer],
                final_norm: vec![1.0; 4],
            },
            BackendKind::Cpu,
        )
        .unwrap()
    }

    fn identity_linear(input: usize, output: usize, with_bias: bool) -> Linear {
        let mut weight = vec![0.0; input * output];
        for index in 0..input.min(output) {
            weight[index * output + index] = 1.0;
        }
        Linear {
            weight,
            bias: with_bias.then(|| vec![0.0; output]),
            in_features: input,
            out_features: output,
        }
    }

    // Deliberately independent scalar oracle: unlike `prefill`, this walks a
    // full causal sequence and computes all attention rows from scratch.
    fn reference_full(runtime: &Qwen2Runtime, embeddings: &[f32], rows: usize) -> Vec<f32> {
        let config = runtime.config();
        let mut keys = vec![Vec::<f32>::new(); config.num_layers];
        let mut values = vec![Vec::<f32>::new(); config.num_layers];
        let mut output = Vec::with_capacity(rows * config.hidden_size);
        for position in 0..rows {
            let d = config.hidden_size;
            let mut hidden = embeddings[position * d..(position + 1) * d].to_vec();
            for layer_index in 0..config.num_layers {
                let layer = &runtime.weights.layers[layer_index];
                let normed = reference_rms(&hidden, &layer.input_norm, config.rms_norm_eps);
                let mut q = reference_linear(&layer.q, &normed);
                let mut k = reference_linear(&layer.k, &normed);
                let v = reference_linear(&layer.v, &normed);
                reference_rope(&mut q, position, config.rope_theta, config.head_dim());
                reference_rope(&mut k, position, config.rope_theta, config.head_dim());
                keys[layer_index].extend_from_slice(&k);
                values[layer_index].extend_from_slice(&v);
                let context = keys[layer_index].len() / config.kv_width();
                let mut attended = vec![0.0; d];
                for head in 0..config.num_attention_heads {
                    let kv_head = head / (config.num_attention_heads / config.num_key_value_heads);
                    let q_start = head * config.head_dim();
                    let mut scores = Vec::with_capacity(context);
                    for row in 0..context {
                        let mut score = 0.0;
                        for component in 0..config.head_dim() {
                            score += q[q_start + component]
                                * keys[layer_index][row * config.kv_width()
                                    + kv_head * config.head_dim()
                                    + component];
                        }
                        scores.push(score / (config.head_dim() as f32).sqrt());
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut probs: Vec<f32> =
                        scores.iter().map(|score| (*score - max).exp()).collect();
                    let denominator: f32 = probs.iter().sum();
                    for prob in &mut probs {
                        *prob /= denominator;
                    }
                    for component in 0..config.head_dim() {
                        attended[q_start + component] = (0..context)
                            .map(|row| {
                                probs[row]
                                    * values[layer_index][row * config.kv_width()
                                        + kv_head * config.head_dim()
                                        + component]
                            })
                            .sum();
                    }
                }
                let projected = reference_linear(&layer.o, &attended);
                for (dst, src) in hidden.iter_mut().zip(projected) {
                    *dst += src;
                }
                let normed = reference_rms(&hidden, &layer.post_norm, config.rms_norm_eps);
                let gate = reference_linear(&layer.gate, &normed);
                let up = reference_linear(&layer.up, &normed);
                let activated: Vec<f32> = gate
                    .into_iter()
                    .zip(up)
                    .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                    .collect();
                let projected = reference_linear(&layer.down, &activated);
                for (dst, src) in hidden.iter_mut().zip(projected) {
                    *dst += src;
                }
            }
            output.extend(reference_rms(
                &hidden,
                &runtime.weights.final_norm,
                config.rms_norm_eps,
            ));
        }
        output
    }

    fn reference_linear(linear: &Linear, input: &[f32]) -> Vec<f32> {
        (0..linear.out_features)
            .map(|out| {
                let mut value = linear.bias.as_ref().map_or(0.0, |bias| bias[out]);
                for (index, &item) in input.iter().enumerate() {
                    value += item * linear.weight[index * linear.out_features + out];
                }
                value
            })
            .collect()
    }

    fn reference_rms(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let mean = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
        let inverse = (mean + eps).sqrt().recip();
        input
            .iter()
            .zip(weight)
            .map(|(value, weight)| value * inverse * weight)
            .collect()
    }

    fn reference_rope(values: &mut [f32], position: usize, theta: f32, head_dim: usize) {
        let half = head_dim / 2;
        for head in values.chunks_exact_mut(head_dim) {
            let before = head.to_vec();
            for pair in 0..half {
                let angle = position as f32 / theta.powf((2 * pair) as f32 / head_dim as f32);
                let (sin, cos) = angle.sin_cos();
                head[pair] = before[pair] * cos - before[half + pair] * sin;
                head[half + pair] = before[half + pair] * cos + before[pair] * sin;
            }
        }
    }
}
