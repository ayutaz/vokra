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

    /// Authenticated Realtime text Qwen2 axes.
    #[must_use]
    pub(crate) const fn vibevoice_realtime_text() -> Self {
        Self {
            hidden_size: 896,
            vocab_size: 151_936,
            num_layers: 4,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            intermediate_size: 4_864,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1.0e-6,
            max_position_embeddings: 8_192,
        }
    }

    /// Authenticated Realtime TTS Qwen2 axes.
    #[must_use]
    pub(crate) const fn vibevoice_realtime_tts() -> Self {
        Self {
            num_layers: 20,
            ..Self::vibevoice_realtime_text()
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

#[derive(Debug, Clone, PartialEq)]
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

/// One caller-owned Qwen2 KV-cache layer in the runtime's native layout.
///
/// Both buffers are flattened row-major `[position, kv-head, head-dim]`
/// values; the key rows must already have RoPE applied, and the value rows are
/// the corresponding runtime values.  This deliberately does not accept or
/// reinterpret a framework cache layout such as PyTorch `DynamicCache`
/// `[batch, kv-head, position, head-dim]`.  Since these are unannotated f32
/// slices, a different ordering with the same element count is
/// indistinguishable from a valid snapshot; the caller must satisfy this
/// native-layout contract.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // consumed by the authenticated Realtime prefill branches
pub(crate) struct Qwen2KvCacheLayer<'a> {
    /// Post-RoPE keys in `[position, kv-head, head-dim]` order.
    pub(crate) keys: &'a [f32],
    /// Values in `[position, kv-head, head-dim]` order.
    pub(crate) values: &'a [f32],
}

/// Caller-owned snapshot for transactional Qwen2 KV-cache import.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // consumed by the authenticated Realtime prefill branches
pub(crate) struct Qwen2KvCacheSnapshot<'a> {
    /// Number of cached positions represented by every layer.
    pub(crate) position: usize,
    /// Exactly one layer entry per configured Qwen2 decoder layer.
    pub(crate) layers: &'a [Qwen2KvCacheLayer<'a>],
}

/// Tied Qwen2 weights bound to the fixed VibeVoice tensor layout.
#[derive(Debug)]
pub(crate) struct Qwen2Weights {
    config: Qwen2RuntimeConfig,
    embedding: Arc<Vec<f32>>,
    layers: Vec<Layer>,
    final_norm: Option<Vec<f32>>,
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
        Self::from_gguf_section(file, "model.language_model", config, true, None)
    }

    /// Loads one of the authenticated Realtime Qwen2 roles.
    ///
    /// Realtime's text stack is constructed with an identity final norm,
    /// while its TTS stack has the regular Qwen2 RMSNorm.  The role prefix
    /// and this policy are explicit so a 1.5B tensor cannot be accepted by
    /// accident and the text path cannot gain an unrequested normalization.
    pub(crate) fn from_gguf_section(
        file: &GgufFile,
        role_prefix: &str,
        config: Qwen2RuntimeConfig,
        with_final_norm: bool,
        shared_embedding: Option<Arc<Vec<f32>>>,
    ) -> Result<Self> {
        if !matches!(
            role_prefix,
            "model.language_model" | "model.tts_language_model"
        ) {
            return Err(VokraError::InvalidArgument(
                "vibevoice Qwen2 role prefix is not an authenticated Realtime role".to_owned(),
            ));
        }
        let expected_config = if role_prefix == "model.language_model" {
            Qwen2RuntimeConfig::vibevoice_realtime_text()
        } else {
            Qwen2RuntimeConfig::vibevoice_realtime_tts()
        };
        let realtime_role = config == expected_config
            && with_final_norm == (role_prefix == "model.tts_language_model");
        let legacy_role = role_prefix == "model.language_model"
            && config == Qwen2RuntimeConfig::vibevoice_1_5b()
            && with_final_norm
            && shared_embedding.is_none();
        if !realtime_role && !legacy_role {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice Qwen2 role `{role_prefix}` has an unauthenticated axes/final-norm policy"
            )));
        }
        config.validate()?;
        let embedding = match shared_embedding {
            Some(embedding) => {
                require_tensor_shape(
                    file,
                    "vibevoice Qwen2",
                    &format!("{role_prefix}.embed_tokens.weight"),
                    &[config.vocab_size, config.hidden_size],
                )?;
                embedding
            }
            None => Arc::new(load_raw(
                file,
                &format!("{role_prefix}.embed_tokens.weight"),
                &[config.vocab_size, config.hidden_size],
            )?),
        };
        let final_norm = if with_final_norm {
            Some(load_raw(
                file,
                &format!("{role_prefix}.norm.weight"),
                &[config.hidden_size],
            )?)
        } else {
            if file
                .tensor_info(&format!("{role_prefix}.norm.weight"))
                .is_some()
            {
                return Err(VokraError::ModelLoad(format!(
                    "vibevoice Qwen2 role `{role_prefix}` unexpectedly carries a final norm"
                )));
            }
            None
        };
        let mut layers = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            let prefix = format!("{role_prefix}.layers.{index}");
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

    /// Binds one authenticated Realtime Qwen2 role on the selected backend.
    pub(crate) fn from_gguf_section_with_backend(
        file: &GgufFile,
        role_prefix: &str,
        config: Qwen2RuntimeConfig,
        backend: BackendKind,
        with_final_norm: bool,
    ) -> Result<Self> {
        Self::from_gguf_section_with_backend_and_embedding(
            file,
            role_prefix,
            config,
            backend,
            with_final_norm,
            None,
        )
    }

    /// Same as [`Self::from_gguf_section_with_backend`] while reusing the
    /// authenticated base embedding.  The Realtime TTS forward calls the
    /// base model's embedding lookup, so loading a second 151936x896 table
    /// would be both unnecessary and an avoidable resident-memory cost.
    pub(crate) fn from_gguf_section_with_backend_and_embedding(
        file: &GgufFile,
        role_prefix: &str,
        config: Qwen2RuntimeConfig,
        backend: BackendKind,
        with_final_norm: bool,
        shared_embedding: Option<Arc<Vec<f32>>>,
    ) -> Result<Self> {
        Self::new(
            Qwen2Weights::from_gguf_section(
                file,
                role_prefix,
                config,
                with_final_norm,
                shared_embedding,
            )?,
            backend,
        )
    }

    pub(crate) fn shared_embedding(&self) -> Arc<Vec<f32>> {
        Arc::clone(&self.weights.embedding)
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

    /// Returns the current number of committed KV-cache positions.
    pub(crate) const fn position(&self) -> usize {
        self.position
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

    /// Imports a caller-owned post-RoPE KV snapshot transactionally.
    ///
    /// The snapshot must contain exactly the configured layer count, and each
    /// layer must contain equal key/value buffers of exactly
    /// `position * (num_key_value_heads * head_dim)` elements in this
    /// runtime's flattened `[position, kv-head, head-dim]` row layout.  The
    /// method never transposes, batches, or otherwise guesses a framework
    /// cache layout.  Because the buffers carry no dimension/order metadata,
    /// a caller-supplied layout with the same element count cannot be detected
    /// here; the native-layout/post-RoPE contract is explicit at this boundary.
    /// All inputs are validated before the runtime cache or position is
    /// replaced; an error therefore leaves the existing generation state
    /// unchanged.  The caller retains ownership of the input buffers, which
    /// are copied only after validation succeeds.
    #[allow(dead_code)] // consumed by the authenticated Realtime prefill branches
    pub(crate) fn import_kv_cache_snapshot(
        &mut self,
        snapshot: Qwen2KvCacheSnapshot<'_>,
    ) -> Result<()> {
        let config = self.config();
        if snapshot.layers.len() != config.num_layers {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice Qwen2 KV snapshot layer count {} does not match configured {}",
                snapshot.layers.len(),
                config.num_layers
            )));
        }
        if snapshot.position > config.max_position_embeddings {
            return Err(VokraError::InvalidArgument(
                "vibevoice Qwen2 KV snapshot position exceeds max_position_embeddings".to_owned(),
            ));
        }
        let kv_width = config
            .num_key_value_heads
            .checked_mul(config.head_dim())
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "vibevoice Qwen2 KV snapshot kv width overflows usize".to_owned(),
                )
            })?;
        let expected_len = snapshot.position.checked_mul(kv_width).ok_or_else(|| {
            VokraError::InvalidArgument(
                "vibevoice Qwen2 KV snapshot position/width product overflows usize".to_owned(),
            )
        })?;

        // Validate every layer before allocating or mutating self.  Length
        // checks establish the flat buffer shape only; they cannot identify a
        // different ordering with the same element count, so ordering remains
        // the caller's explicit native-layout contract.
        for (layer_index, layer) in snapshot.layers.iter().enumerate() {
            if layer.keys.len() != layer.values.len() {
                return Err(VokraError::InvalidArgument(format!(
                    "vibevoice Qwen2 KV snapshot layer {layer_index} key/value lengths differ"
                )));
            }
            if layer.keys.len() != expected_len {
                return Err(VokraError::InvalidArgument(format!(
                    "vibevoice Qwen2 KV snapshot layer {layer_index} has {} elements, expected {expected_len}",
                    layer.keys.len()
                )));
            }
            if layer.keys.iter().any(|value| !value.is_finite())
                || layer.values.iter().any(|value| !value.is_finite())
            {
                return Err(VokraError::InvalidArgument(format!(
                    "vibevoice Qwen2 KV snapshot layer {layer_index} contains non-finite values"
                )));
            }
        }

        let staged = snapshot
            .layers
            .iter()
            .map(|layer| LayerCache {
                keys: layer.keys.to_vec(),
                values: layer.values.to_vec(),
            })
            .collect();
        self.cache = staged;
        self.position = snapshot.position;
        Ok(())
    }

    /// Imports a native-layout snapshot supplied as borrowed key/value pairs.
    ///
    /// This narrow bridge keeps the authenticated Qwen2 snapshot validator as
    /// the single implementation while allowing sibling composite modules to
    /// keep their own pair/snapshot types.  It never infers or transposes a
    /// framework layout; the slices must already be post-RoPE and flattened as
    /// `[position, kv-head, head-dim]`.
    #[allow(dead_code)] // consumed by the staged Realtime four-output bridge
    pub(crate) fn import_kv_cache_snapshot_parts(
        &mut self,
        position: usize,
        layers: &[(&[f32], &[f32])],
    ) -> Result<()> {
        let snapshot_layers: Vec<_> = layers
            .iter()
            .map(|layer| Qwen2KvCacheLayer {
                keys: layer.0,
                values: layer.1,
            })
            .collect();
        self.import_kv_cache_snapshot(Qwen2KvCacheSnapshot {
            position,
            layers: &snapshot_layers,
        })
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
        self.finalize_hidden(&compute, hidden)
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
        self.finalize_hidden_rows(&compute, hidden, rows, d)
    }

    fn finalize_hidden(&self, compute: &Compute, hidden: Vec<f32>) -> Result<Vec<f32>> {
        match self.weights.final_norm.as_deref() {
            Some(weight) => rms(compute, &hidden, weight, self.config().rms_norm_eps),
            None => {
                finite("Qwen2 final hidden", &hidden)?;
                Ok(hidden)
            }
        }
    }

    fn finalize_hidden_rows(
        &self,
        compute: &Compute,
        hidden: Vec<f32>,
        rows: usize,
        width: usize,
    ) -> Result<Vec<f32>> {
        match self.weights.final_norm.as_deref() {
            Some(weight) => rms_rows(
                compute,
                &hidden,
                weight,
                rows,
                width,
                self.config().rms_norm_eps,
            ),
            None => {
                finite("Qwen2 final hidden rows", &hidden)?;
                Ok(hidden)
            }
        }
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
        || weights
            .final_norm
            .as_ref()
            .is_some_and(|norm| norm.len() != config.hidden_size)
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
/// Small model-free runtime shared by the Qwen2 and Realtime language tests.
pub(crate) fn test_fixture_runtime() -> Qwen2Runtime {
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
        q: test_identity_linear(4, 4, true),
        k: test_identity_linear(4, 2, true),
        v: test_identity_linear(4, 2, true),
        o: test_identity_linear(4, 4, false),
        input_norm: vec![1.0; 4],
        post_norm: vec![1.0; 4],
        gate: test_identity_linear(4, 8, false),
        up: test_identity_linear(4, 8, false),
        down: test_identity_linear(8, 4, false),
    };
    let embedding = (0..config.vocab_size * config.hidden_size)
        .map(|index| (index % config.hidden_size) as f32 * 0.1 + 0.1)
        .collect();
    Qwen2Runtime::new(
        Qwen2Weights {
            config,
            embedding: Arc::new(embedding),
            layers: vec![layer],
            final_norm: Some(vec![1.0; 4]),
        },
        BackendKind::Cpu,
    )
    .expect("test Qwen2 fixture must bind")
}

#[cfg(test)]
fn test_identity_linear(input: usize, output: usize, with_bias: bool) -> Linear {
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

#[cfg(test)]
impl Qwen2Runtime {
    /// Shares the module-level model-free fixture with sibling tests.
    pub(crate) fn test_fixture_runtime() -> Self {
        test_fixture_runtime()
    }

    /// Exposes only the cache position for model-free composite tests.
    pub(crate) fn test_position(&self) -> usize {
        self.position()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::test_fixture_runtime as fixture_runtime;
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
    fn kv_snapshot_import_matches_prefilled_runtime_on_next_step() {
        let mut original = fixture_runtime();
        let prompt = [0.1_f32, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1];
        original.prefill_embeddings(&prompt, 2).unwrap();
        let snapshot_layers: Vec<_> = original
            .cache
            .iter()
            .map(|layer| Qwen2KvCacheLayer {
                keys: &layer.keys,
                values: &layer.values,
            })
            .collect();
        let snapshot = Qwen2KvCacheSnapshot {
            position: original.position,
            layers: &snapshot_layers,
        };
        let mut imported = original.fork_empty_cache();
        imported.import_kv_cache_snapshot(snapshot).unwrap();

        let next = [0.6_f32, -0.5, 0.4, -0.3];
        let expected = original.step_embedding(&next).unwrap();
        let actual = imported.step_embedding(&next).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(imported.position, original.position);
        assert_eq!(imported.cache, original.cache);
    }

    #[test]
    fn kv_snapshot_rejects_invalid_inputs_without_mutating_state() {
        let mut runtime = fixture_runtime();
        runtime
            .prefill_embeddings(&[0.1, 0.2, 0.3, 0.4], 1)
            .unwrap();

        let empty_layers: [Qwen2KvCacheLayer<'_>; 0] = [];
        assert_snapshot_rejected(
            &mut runtime,
            Qwen2KvCacheSnapshot {
                position: 1,
                layers: &empty_layers,
            },
        );

        let valid_keys = [0.1_f32, 0.2];
        let valid_values = [0.3_f32, 0.4];
        let valid_layers = [Qwen2KvCacheLayer {
            keys: &valid_keys,
            values: &valid_values,
        }];
        let max_position = runtime.config().max_position_embeddings;
        assert_snapshot_rejected(
            &mut runtime,
            Qwen2KvCacheSnapshot {
                position: max_position + 1,
                layers: &valid_layers,
            },
        );

        let wrong_shape_layers = [Qwen2KvCacheLayer {
            keys: &[0.1_f32],
            values: &[0.2_f32],
        }];
        assert_snapshot_rejected(
            &mut runtime,
            Qwen2KvCacheSnapshot {
                position: 1,
                layers: &wrong_shape_layers,
            },
        );

        let mismatched_layers = [Qwen2KvCacheLayer {
            keys: &[0.1_f32],
            values: &valid_values,
        }];
        assert_snapshot_rejected(
            &mut runtime,
            Qwen2KvCacheSnapshot {
                position: 1,
                layers: &mismatched_layers,
            },
        );

        let nonfinite_keys = [f32::NAN, 0.2_f32];
        let nonfinite_layers = [Qwen2KvCacheLayer {
            keys: &nonfinite_keys,
            values: &valid_values,
        }];
        assert_snapshot_rejected(
            &mut runtime,
            Qwen2KvCacheSnapshot {
                position: 1,
                layers: &nonfinite_layers,
            },
        );
    }

    fn assert_snapshot_rejected(runtime: &mut Qwen2Runtime, snapshot: Qwen2KvCacheSnapshot<'_>) {
        let before_cache = runtime.cache.clone();
        let before_position = runtime.position;
        assert!(runtime.import_kv_cache_snapshot(snapshot).is_err());
        assert_eq!(runtime.position, before_position);
        assert_eq!(runtime.cache, before_cache);
    }

    #[test]
    fn cloned_cache_branches_append_without_mutating_each_other() {
        let mut runtime = fixture_runtime();
        runtime.step_embedding(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        let mut branch = runtime.clone();
        branch.step_embedding(&[0.4, 0.3, 0.2, 0.1]).unwrap();

        assert_eq!(runtime.position, 1);
        assert_eq!(branch.position, 2);
        assert_eq!(runtime.cache[0].keys.len(), 2);
        assert_eq!(branch.cache[0].keys.len(), 4);
    }

    #[test]
    fn independently_prefilled_branches_keep_prompt_specific_context() {
        let mut positive = fixture_runtime();
        let mut negative = fixture_runtime();
        positive
            .prefill_embeddings(&[0.1, 0.2, 0.3, 0.4], 1)
            .unwrap();
        negative
            .prefill_embeddings(&[0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5], 2)
            .unwrap();

        positive.step_embedding(&[0.5, 0.6, 0.7, 0.8]).unwrap();
        negative.step_embedding(&[0.5, 0.6, 0.7, 0.8]).unwrap();

        assert_eq!(positive.position, 2);
        assert_eq!(negative.position, 3);
        assert_eq!(positive.cache[0].keys.len(), 4);
        assert_eq!(negative.cache[0].keys.len(), 6);
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
            let final_norm = runtime
                .weights
                .final_norm
                .as_deref()
                .expect("legacy 1.5B Qwen2 oracle requires a final norm");
            output.extend(reference_rms(&hidden, final_norm, config.rms_norm_eps));
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
