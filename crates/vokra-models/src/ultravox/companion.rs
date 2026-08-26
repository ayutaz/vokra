//! Strict runtime boundary for the separately licensed Llama 3.2 companion.
//!
//! This module binds only a user-acquired conversion of the exact gated
//! `meta-llama/Llama-3.2-1B-Instruct` snapshot. It never downloads, publishes,
//! or merges those weights into the public MIT Ultravox audio artifact.

use std::path::Path;
use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::compliance::{CompliancePolicy, check_weight_license};
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

use super::companion_decoder::UltravoxLlamaDecoderRuntime;
use super::companion_weights::UltravoxLlamaMappedDescriptors;
use super::projector::UltravoxAudioEmbeddings;

/// Architecture tag of the separately acquired text companion.
pub const COMPANION_ARCH: &str = "ultravox_llama_companion";
/// Exact model identity emitted by the strict converter.
pub const COMPANION_MODEL_NAME: &str = "meta-llama-3.2-1b-instruct-ultravox-companion";
/// Gated source repository. No runtime path downloads from it.
pub const COMPANION_UPSTREAM_HF: &str = "meta-llama/Llama-3.2-1B-Instruct";
/// Only source revision admitted by this runtime contract.
pub const COMPANION_SOURCE_REVISION: &str = "9213176726f574b556790deb65791e0c5aa438b6";
/// Exact HF model-card license identifier.
pub const COMPANION_LICENSE: &str = "llama3.2";
/// Complete sorted name/shape manifest digest of the 146 BF16 tensors.
pub const COMPANION_MANIFEST_SHA256: &str =
    "7832a30cf077054292c8728a5e04621bfb431369566db282b3ccf1692a4e3712";

const LABEL: &str = "ultravox_llama_companion";
const CATEGORY: &str = "text-llm-companion";
const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const PREFIX: &str = "vokra.ultravox.companion";

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: COMPANION_ARCH,
    model_name: COMPANION_MODEL_NAME,
    model_name_alias: None,
    tensor_count: 146,
    manifest_sha256: [
        0x78, 0x32, 0xa3, 0x0c, 0xf0, 0x77, 0x05, 0x42, 0x92, 0xc8, 0x72, 0x8a, 0x5e, 0x04, 0x62,
        0x1b, 0xfb, 0x43, 0x13, 0x69, 0x56, 0x6d, 0xb2, 0x82, 0xb3, 0xcc, 0xf1, 0x69, 0x2a, 0x4e,
        0x37, 0x12,
    ],
};

/// Every learned operation required by the native Llama 3.2 decoder.
pub const ULTRAVOX_LLAMA_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::RmsNorm,
    HotOp::Silu,
];

/// Runtime axes read from and authenticated against the converter metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UltravoxLlamaConfig {
    pub hidden_size: u32,
    pub n_layer: u32,
    pub n_head: u32,
    pub n_kv_head: u32,
    pub head_dim: u32,
    pub ffn_dim: u32,
    pub vocab_size: u32,
    pub max_positions: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub rope_low_freq_factor: f32,
    pub rope_high_freq_factor: f32,
    pub rope_original_max_positions: u32,
}

impl UltravoxLlamaConfig {
    /// Exact official Llama-3.2-1B-Instruct topology.
    pub const OFFICIAL: Self = Self {
        hidden_size: 2_048,
        n_layer: 16,
        n_head: 32,
        n_kv_head: 8,
        head_dim: 64,
        ffn_dim: 8_192,
        vocab_size: 128_256,
        max_positions: 131_072,
        rms_norm_eps: 1.0e-5,
        rope_theta: 500_000.0,
        rope_factor: 32.0,
        rope_low_freq_factor: 1.0,
        rope_high_freq_factor: 4.0,
        rope_original_max_positions: 8_192,
    };
}

/// Bounded deterministic generation controls over an exact token-id prompt.
///
/// The companion GGUF intentionally contains no tokenizer or generation
/// sidecar. Callers must therefore provide the exact stop-token IDs belonging
/// to the tokenizer that produced the prompt; Vokra never guesses them.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UltravoxGenerationOptions {
    /// Maximum tokens emitted after the complete multimodal prompt.
    pub max_new_tokens: usize,
    /// One or more exact tokenizer IDs that terminate generation.
    pub stop_token_ids: Vec<u32>,
}

impl UltravoxGenerationOptions {
    /// Constructs an explicit deterministic greedy request.
    #[must_use]
    pub fn greedy(max_new_tokens: usize, stop_token_ids: Vec<u32>) -> Self {
        Self {
            max_new_tokens,
            stop_token_ids,
        }
    }

    fn validate(&self, prompt_len: usize, config: UltravoxLlamaConfig) -> Result<()> {
        if self.max_new_tokens == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: max_new_tokens must be greater than zero"
            )));
        }
        if self.stop_token_ids.is_empty() {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: stop_token_ids must name at least one exact tokenizer terminator"
            )));
        }
        if let Some((index, token)) = self
            .stop_token_ids
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token)| *token >= config.vocab_size)
        {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: stop_token_ids[{index}]={token} is outside vocabulary 0..{}",
                config.vocab_size
            )));
        }
        let positions = prompt_len
            .checked_add(self.max_new_tokens - 1)
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "{LABEL}: prompt plus generation position count overflows"
                ))
            })?;
        if positions > config.max_positions as usize {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: prompt {prompt_len} + at most {} forwarded generation rows exceeds max positions {}",
                self.max_new_tokens - 1,
                config.max_positions
            )));
        }
        Ok(())
    }
}

/// Exact greedy token sequence emitted by the companion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UltravoxGeneration {
    /// Every generated token, including the terminating token when observed.
    pub token_ids: Vec<u32>,
    /// The first caller-supplied stop token observed, or `None` at the cap.
    pub stop_token: Option<u32>,
}

/// Strict mmap-backed handle for the user-acquired Llama companion.
pub struct UltravoxLlamaCompanion {
    checkpoint: StrictCheckpoint,
    mapped: Arc<UltravoxLlamaMappedDescriptors>,
    runtime: UltravoxLlamaDecoderRuntime,
    backend: BackendKind,
    source_revision: String,
}

impl std::fmt::Debug for UltravoxLlamaCompanion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UltravoxLlamaCompanion")
            .field("tensor_count", &self.mapped.descriptor_count())
            .field("weight_license", &self.checkpoint.weight_license())
            .field("backend", &self.backend)
            .field("source_revision", &self.source_revision)
            .finish()
    }
}

impl UltravoxLlamaCompanion {
    /// Opens the exact dense companion by mmap under strict policy on CPU.
    pub fn open_mapped(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_mapped_with_policy_and_backend(
            path,
            &CompliancePolicy::strict(),
            BackendKind::Cpu,
        )
    }

    /// Opens, license-gates, validates, and preflights one backend.
    pub fn open_mapped_with_policy_and_backend(
        path: impl AsRef<Path>,
        policy: &CompliancePolicy,
        backend: BackendKind,
    ) -> Result<Self> {
        let file = vokra_mmap::open_gguf(path.as_ref()).map_err(VokraError::from)?;
        Self::from_gguf_mapped_with_policy_and_backend(Arc::new(file), policy, backend)
    }

    /// Strictly binds an already mmap-backed companion artifact.
    pub fn from_gguf_mapped_with_policy_and_backend(
        file: Arc<GgufFile>,
        policy: &CompliancePolicy,
        backend: BackendKind,
    ) -> Result<Self> {
        check_weight_license(&file, policy)?;
        let checkpoint = StrictCheckpoint::bind(&file, SPEC)?;
        require_string(&file, KEY_CATEGORY, CATEGORY)?;
        require_string(&file, chunks::KEY_PROVENANCE_MODEL_ID, COMPANION_MODEL_NAME)?;
        require_string(&file, KEY_UPSTREAM_HF, COMPANION_UPSTREAM_HF)?;
        require_string(&file, chunks::KEY_PROVENANCE_LICENSE, COMPANION_LICENSE)?;
        require_string(
            &file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::ConditionalCommercial.as_str(),
        )?;
        if checkpoint.weight_license() != LicenseClass::ConditionalCommercial {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: expected ConditionalCommercial Llama 3.2 weights, got {:?}",
                checkpoint.weight_license()
            )));
        }

        let source_revision = require_string(
            &file,
            &format!("{PREFIX}.source_revision"),
            COMPANION_SOURCE_REVISION,
        )?
        .to_owned();
        require_string(&file, KEY_UPSTREAM_REVISION, COMPANION_SOURCE_REVISION)?;
        require_hex_string(&file, &format!("{PREFIX}.config_sha256"), 64)?;
        require_string(
            &file,
            &format!("{PREFIX}.tensor_manifest_sha256"),
            COMPANION_MANIFEST_SHA256,
        )?;
        let config = read_config(&file)?;
        if config != UltravoxLlamaConfig::OFFICIAL {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: runtime metadata {config:?} does not match the admitted official topology {:?}",
                UltravoxLlamaConfig::OFFICIAL
            )));
        }
        require_string(&file, &format!("{PREFIX}.hidden_act"), "silu")?;
        require_string(&file, &format!("{PREFIX}.rope.type"), "llama3")?;
        require_bool(&file, &format!("{PREFIX}.tied_embeddings"), true)?;
        require_bool(&file, &format!("{PREFIX}.attention_bias"), false)?;
        require_bool(&file, &format!("{PREFIX}.mlp_bias"), false)?;

        Compute::for_backend(backend, ULTRAVOX_LLAMA_HOT_OPS)?;
        let mapped = Arc::new(UltravoxLlamaMappedDescriptors::bind(file, config)?);
        Ok(Self {
            checkpoint,
            mapped,
            runtime: UltravoxLlamaDecoderRuntime::default(),
            backend,
            source_revision,
        })
    }

    /// Exact metadata-derived Llama topology.
    #[must_use]
    pub fn config(&self) -> UltravoxLlamaConfig {
        self.mapped.config()
    }

    /// Backend preflighted for every decoder hot operation.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Authenticated weight-license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    /// Exact gated source snapshot named by both provenance fields.
    #[must_use]
    pub fn source_revision(&self) -> &str {
        &self.source_revision
    }

    /// Complete number of mapped tensor descriptors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.mapped.descriptor_count()
    }

    /// Greedily generates from an exact pre-tokenized Ultravox prompt.
    ///
    /// `audio_token_start_idx` is the first row that the official processor
    /// reserved for audio. Exactly `audio.frames()` consecutive ordinary token
    /// embeddings are replaced, matching the upstream `inputs_embeds` route.
    /// The prompt's IDs, placeholder expansion and stop IDs remain explicit so
    /// this tokenizer-less companion never downloads or invents sidecars.
    pub fn generate_with_audio_embeddings(
        &self,
        prompt_token_ids: &[u32],
        audio_token_start_idx: usize,
        audio: &UltravoxAudioEmbeddings,
        options: &UltravoxGenerationOptions,
    ) -> Result<UltravoxGeneration> {
        validate_audio_prompt(
            prompt_token_ids,
            audio_token_start_idx,
            audio,
            self.config(),
        )?;
        options.validate(prompt_token_ids.len(), self.config())?;
        super::companion_decoder::generate(
            &self.mapped,
            self.backend,
            &self.runtime,
            prompt_token_ids,
            audio_token_start_idx,
            audio,
            options,
        )
    }

    /// Returns full-vocabulary logits for the first generated position.
    ///
    /// This deterministic parity tap executes the same mapped prefill,
    /// audio-embedding replacement and selected backend as generation.
    pub fn next_token_logits_with_audio_embeddings(
        &self,
        prompt_token_ids: &[u32],
        audio_token_start_idx: usize,
        audio: &UltravoxAudioEmbeddings,
    ) -> Result<Vec<f32>> {
        validate_audio_prompt(
            prompt_token_ids,
            audio_token_start_idx,
            audio,
            self.config(),
        )?;
        super::companion_decoder::next_token_logits(
            &self.mapped,
            self.backend,
            &self.runtime,
            prompt_token_ids,
            audio_token_start_idx,
            audio,
        )
    }
}

fn validate_audio_prompt(
    prompt: &[u32],
    audio_start: usize,
    audio: &UltravoxAudioEmbeddings,
    config: UltravoxLlamaConfig,
) -> Result<()> {
    if prompt.is_empty() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: prompt token IDs are empty"
        )));
    }
    if prompt.len() > config.max_positions as usize {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: prompt length {} exceeds max positions {}",
            prompt.len(),
            config.max_positions
        )));
    }
    if audio.frames() == 0
        || audio.hidden_size() != config.hidden_size as usize
        || audio.values().len() != audio.frames().saturating_mul(audio.hidden_size())
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: projected audio shape is [{},{}] with {} values; decoder requires non-empty [frames,{}]",
            audio.frames(),
            audio.hidden_size(),
            audio.values().len(),
            config.hidden_size
        )));
    }
    let audio_end = audio_start.checked_add(audio.frames()).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{LABEL}: audio placeholder span overflows usize"))
    })?;
    if audio_end > prompt.len() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: audio rows {audio_start}..{audio_end} exceed prompt length {}",
            prompt.len()
        )));
    }
    if let Some((index, token)) = prompt
        .iter()
        .copied()
        .enumerate()
        .find(|(_, token)| *token >= config.vocab_size)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: prompt token {token} at row {index} is outside vocabulary 0..{}",
            config.vocab_size
        )));
    }
    if let Some((index, value)) = audio
        .values()
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: projected audio contains non-finite value {value} at index {index}"
        )));
    }
    Ok(())
}

fn read_config(file: &GgufFile) -> Result<UltravoxLlamaConfig> {
    Ok(UltravoxLlamaConfig {
        hidden_size: required_u32(file, &format!("{PREFIX}.hidden_size"))?,
        n_layer: required_u32(file, &format!("{PREFIX}.n_layer"))?,
        n_head: required_u32(file, &format!("{PREFIX}.n_head"))?,
        n_kv_head: required_u32(file, &format!("{PREFIX}.n_kv_head"))?,
        head_dim: required_u32(file, &format!("{PREFIX}.head_dim"))?,
        ffn_dim: required_u32(file, &format!("{PREFIX}.ffn_dim"))?,
        vocab_size: required_u32(file, &format!("{PREFIX}.vocab_size"))?,
        max_positions: required_u32(file, &format!("{PREFIX}.max_positions"))?,
        rms_norm_eps: required_f32(file, &format!("{PREFIX}.rms_norm_eps"))?,
        rope_theta: required_f32(file, &format!("{PREFIX}.rope_theta"))?,
        rope_factor: required_f32(file, &format!("{PREFIX}.rope.factor"))?,
        rope_low_freq_factor: required_f32(file, &format!("{PREFIX}.rope.low_freq_factor"))?,
        rope_high_freq_factor: required_f32(file, &format!("{PREFIX}.rope.high_freq_factor"))?,
        rope_original_max_positions: required_u32(
            file,
            &format!("{PREFIX}.rope.original_max_positions"),
        )?,
    })
}

fn required_string<'a>(file: &'a GgufFile, key: &str, expected: &str) -> Result<&'a str> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}` is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(actual.expect("checked Some above"))
}

fn required_u32(file: &GgufFile, key: &str) -> Result<u32> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_u64);
    actual
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            VokraError::ModelLoad(format!("{LABEL}: `{key}` is {actual:?}, expected u32"))
        })
}

fn required_f32(file: &GgufFile, key: &str) -> Result<f32> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_f64);
    actual
        .filter(|value| value.is_finite() && *value >= f32::MIN as f64 && *value <= f32::MAX as f64)
        .map(|value| value as f32)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "{LABEL}: `{key}` is {actual:?}, expected finite f32"
            ))
        })
}

fn require_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_bool);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}` is {actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_hex_string(file: &GgufFile, key: &str, len: usize) -> Result<()> {
    let value = file
        .get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("{LABEL}: missing/non-string `{key}`")))?;
    if value.len() != len || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}` must be exactly {len} hexadecimal characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strict_checkpoint::sha256_bytes;

    fn manifest_digest(config: UltravoxLlamaConfig) -> [u8; 32] {
        let mut contract = super::super::companion_weights::tensor_contract(config);
        contract.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        let mut canonical = Vec::new();
        for (name, dimensions) in contract {
            canonical.extend_from_slice(name.as_bytes());
            canonical.push(0);
            canonical.extend_from_slice(&(dimensions.len() as u64).to_le_bytes());
            for dimension in dimensions {
                canonical.extend_from_slice(&dimension.to_le_bytes());
            }
        }
        sha256_bytes(&canonical)
    }

    fn hex(bytes: &[u8; 32]) -> String {
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(64);
        for byte in bytes {
            output.push(char::from(DIGITS[(byte >> 4) as usize]));
            output.push(char::from(DIGITS[(byte & 0x0f) as usize]));
        }
        output
    }

    #[test]
    fn manifest_and_source_snapshot_are_pinned() {
        assert_eq!(hex(&SPEC.manifest_sha256), COMPANION_MANIFEST_SHA256);
        assert_eq!(
            manifest_digest(UltravoxLlamaConfig::OFFICIAL),
            SPEC.manifest_sha256
        );
        assert_eq!(COMPANION_SOURCE_REVISION.len(), 40);
        assert!(
            COMPANION_SOURCE_REVISION
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        );
    }

    #[test]
    fn official_topology_has_valid_grouped_query_axes() {
        let config = UltravoxLlamaConfig::OFFICIAL;
        assert_eq!(config.hidden_size, config.n_head * config.head_dim);
        assert_eq!(config.n_head % config.n_kv_head, 0);
        assert_eq!(config.rope_original_max_positions, 8_192);
        assert!(config.rope_high_freq_factor > config.rope_low_freq_factor);
    }

    #[test]
    fn audio_prompt_requires_an_exact_consecutive_replacement_span() {
        let config = UltravoxLlamaConfig::OFFICIAL;
        let audio = UltravoxAudioEmbeddings {
            values: vec![0.0; 2 * config.hidden_size as usize],
            frames: 2,
            hidden_size: config.hidden_size as usize,
        };
        validate_audio_prompt(&[1, 2, 3, 4], 1, &audio, config).expect("valid span");
        assert!(validate_audio_prompt(&[1, 2], 1, &audio, config).is_err());
        assert!(validate_audio_prompt(&[], 0, &audio, config).is_err());
    }

    #[test]
    fn generation_requires_explicit_bounded_stop_ids() {
        let config = UltravoxLlamaConfig::OFFICIAL;
        assert!(
            UltravoxGenerationOptions::greedy(0, vec![1])
                .validate(4, config)
                .is_err()
        );
        assert!(
            UltravoxGenerationOptions::greedy(1, vec![])
                .validate(4, config)
                .is_err()
        );
        assert!(
            UltravoxGenerationOptions::greedy(1, vec![config.vocab_size])
                .validate(4, config)
                .is_err()
        );
        UltravoxGenerationOptions::greedy(4, vec![1])
            .validate(4, config)
            .expect("bounded explicit generation");
    }
}
