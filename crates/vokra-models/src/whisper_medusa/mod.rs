//! Native `aiola/whisper-medusa-v1` runtime.
//!
//! The released model is Whisper-large-v2 plus eleven `base_head` residual
//! modules.  Module 0 produces the current-token logits and is therefore part
//! of the canonical non-speculative forward; modules 1–10 propose future
//! tokens for Medusa tree verification.  This binder implements the exact
//! module-0 path and never substitutes plain Whisper logits.
//!
//! The optional accelerated draft/verify/accept API remains explicit until
//! its tree-attention driver is implemented.  Ordinary `AsrEngine`
//! transcription is nevertheless the official model forward, not a fallback.

use std::sync::Arc;

use vokra_core::engines::AsrEngine;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, GgufValueType, chunks};
use vokra_core::tasks::Transcription;
use vokra_core::{BackendKind, LicenseClass, Result, VokraError};

use crate::whisper::decoder::ResidualSiluLogitsAdapter;
use crate::whisper::{WhisperAsr, WhisperTokenizer};

/// Canonical GGUF architecture tag.
pub const ARCH: &str = "whisper-medusa-v1";
/// Canonical model name.
pub const NAME: &str = "whisper-medusa-v1";
/// Runtime task category.
pub const CATEGORY: &str = "asr";
/// Canonical upstream Hugging Face repository.
pub const UPSTREAM_HF: &str = "aiola/whisper-medusa-v1";
/// SPDX license stamped by the canonical converter.
pub const CONVERTER_DEFAULT_LICENSE: &str = "MIT";

const UPSTREAM_REVISION: &str = "6ea7c2f47658cfc7f9c8d1c158a9fbdb33458462";
const UPSTREAM_SOURCE_REVISION: &str = "19819c37ab15db6e68826e406614a2c86fbb946e";
const CONFIG_SHA256: &str = "16346762b14c116eeda12b48f20e2281b327a11b516f8b004ce065fcb1450186";
const CHECKPOINT_SHA256: &str = "ec634d5ece33a8d634ed2e188c7bfbde7adab4410932b8fa6c20440836a423f3";

const KEY_REVISION: &str = "vokra.medusa.revision";
const KEY_SOURCE_REVISION: &str = "vokra.medusa.source_revision";
const KEY_CONFIG_SHA256: &str = "vokra.medusa.config_sha256";
const KEY_CHECKPOINT_SHA256: &str = "vokra.medusa.checkpoint_sha256";
const KEY_MEDUSA_NUM_HEADS: &str = "vokra.medusa.num_heads";
const KEY_MEDUSA_MODULE_COUNT: &str = "vokra.medusa.module_count";
const KEY_MEDUSA_NUM_LAYERS: &str = "vokra.medusa.num_layers";
const KEY_MEDUSA_HIDDEN_SIZE: &str = "vokra.medusa.hidden_size";
const KEY_MEDUSA_HEADS_TYPE: &str = "vokra.medusa.heads_type";
const KEY_MEDUSA_CHOICES: &str = "vokra.medusa.choices";
const KEY_MEDUSA_INIT_FROM_PROJ: &str = "vokra.medusa.init_from_proj";
const KEY_OUTPUT_WHISPER_ORIGINAL: &str = "vokra.medusa.output_whisper_original";

const MEDUSA_PREFIX: &str = "medusa_heads.";
const EXPECTED_NUM_HEADS: usize = 10;
const EXPECTED_MODULE_COUNT: usize = 11;
const EXPECTED_NUM_LAYERS: usize = 1;
const EXPECTED_HIDDEN_SIZE: usize = 1280;
const EXPECTED_CHOICES: [u32; 11] = [1; 11];

#[derive(Debug, Clone, PartialEq, Eq)]
/// Strict geometry and behavior contract read from `vokra.medusa.*`.
pub struct MedusaConfig {
    /// Number of speculative future-token heads.  The checkpoint has one
    /// additional module 0 for current-token logits.
    pub num_heads: usize,
    /// Total residual modules, including module 0.
    pub module_count: usize,
    /// Residual blocks within each module.
    pub num_layers: usize,
    /// Input/output width of every residual block.
    pub hidden_size: usize,
    /// Upstream head family (`base_head` for v1).
    pub heads_type: String,
    /// Upstream candidate count per tree depth.
    pub choices: Vec<u32>,
    /// Whether the official checkpoint initialized heads from projection.
    pub init_from_proj: bool,
    /// Whether plain Whisper logits are additionally emitted.
    pub output_whisper_original: bool,
}

impl MedusaConfig {
    /// Reads and validates the complete pinned metadata group.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_string(file, KEY_REVISION, UPSTREAM_REVISION)?;
        require_string(file, KEY_SOURCE_REVISION, UPSTREAM_SOURCE_REVISION)?;
        require_string(file, KEY_CONFIG_SHA256, CONFIG_SHA256)?;
        require_string(file, KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256)?;
        let config = Self {
            num_heads: require_usize(file, KEY_MEDUSA_NUM_HEADS)?,
            module_count: require_usize(file, KEY_MEDUSA_MODULE_COUNT)?,
            num_layers: require_usize(file, KEY_MEDUSA_NUM_LAYERS)?,
            hidden_size: require_usize(file, KEY_MEDUSA_HIDDEN_SIZE)?,
            heads_type: file
                .get(KEY_MEDUSA_HEADS_TYPE)
                .and_then(GgufMetadataValue::as_str)
                .ok_or_else(|| missing_or_wrong(KEY_MEDUSA_HEADS_TYPE, "string"))?
                .to_owned(),
            choices: require_u32_array(file, KEY_MEDUSA_CHOICES)?,
            init_from_proj: require_bool(file, KEY_MEDUSA_INIT_FROM_PROJ)?,
            output_whisper_original: require_bool(file, KEY_OUTPUT_WHISPER_ORIGINAL)?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.num_heads != EXPECTED_NUM_HEADS
            || self.module_count != EXPECTED_MODULE_COUNT
            || self.num_layers != EXPECTED_NUM_LAYERS
            || self.hidden_size != EXPECTED_HIDDEN_SIZE
            || self.heads_type != "base_head"
            || self.choices != EXPECTED_CHOICES
            || !self.init_from_proj
            || self.output_whisper_original
        {
            return Err(VokraError::ModelLoad(format!(
                "whisper-medusa metadata does not match pinned v1: got {self:?}; expected \
                 num_heads=10, module_count=11, num_layers=1, hidden_size=1280, \
                 heads_type=base_head, choices=[1;11], init_from_proj=true, \
                 output_whisper_original=false"
            )));
        }
        Ok(())
    }
}

fn missing_or_wrong(key: &str, expected: &str) -> VokraError {
    VokraError::ModelLoad(format!(
        "whisper-medusa: GGUF metadata `{key}` is missing or is not {expected}"
    ))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let found = file
        .get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| missing_or_wrong(key, "a string"))?;
    if found != expected {
        return Err(VokraError::ModelLoad(format!(
            "whisper-medusa: `{key}` is `{found}`, expected pinned `{expected}`"
        )));
    }
    Ok(())
}

fn require_usize(file: &GgufFile, key: &str) -> Result<usize> {
    let value = file
        .get(key)
        .and_then(GgufMetadataValue::as_u64)
        .ok_or_else(|| missing_or_wrong(key, "an unsigned integer"))?;
    usize::try_from(value).map_err(|_| missing_or_wrong(key, "representable as usize"))
}

fn require_bool(file: &GgufFile, key: &str) -> Result<bool> {
    match file.get(key) {
        Some(GgufMetadataValue::Bool(value)) => Ok(*value),
        _ => Err(missing_or_wrong(key, "a bool")),
    }
}

fn require_u32_array(file: &GgufFile, key: &str) -> Result<Vec<u32>> {
    let array = file
        .get(key)
        .and_then(GgufMetadataValue::as_array)
        .ok_or_else(|| missing_or_wrong(key, "a U32 array"))?;
    if array.element_type != GgufValueType::U32 {
        return Err(missing_or_wrong(key, "a U32 array"));
    }
    array
        .values
        .iter()
        .map(|value| match value {
            GgufMetadataValue::U32(value) => Ok(*value),
            _ => Err(missing_or_wrong(key, "a homogeneous U32 array")),
        })
        .collect()
}

fn validate_head_manifest(file: &GgufFile, config: &MedusaConfig) -> Result<()> {
    if file
        .tensors()
        .iter()
        .any(|tensor| tensor.name.starts_with("medusa_head."))
    {
        return Err(VokraError::ModelLoad(
            "whisper-medusa: singular `medusa_head.*` is not part of the official v1 \
             checkpoint; refusing an ambiguous scaffold artifact"
                .into(),
        ));
    }

    let head_tensors: Vec<_> = file
        .tensors()
        .iter()
        .filter(|tensor| tensor.name.starts_with(MEDUSA_PREFIX))
        .collect();
    if head_tensors.len() != config.module_count * 2 {
        return Err(VokraError::ModelLoad(format!(
            "whisper-medusa: expected {} head tensors (11 modules × weight+bias), found {}",
            config.module_count * 2,
            head_tensors.len(),
        )));
    }

    for index in 0..config.module_count {
        let weight = format!("{MEDUSA_PREFIX}{index}.0.linear.weight");
        let bias = format!("{MEDUSA_PREFIX}{index}.0.linear.bias");
        require_tensor_shape(file, &weight, &[config.hidden_size, config.hidden_size])?;
        require_tensor_shape(file, &bias, &[config.hidden_size])?;
    }
    Ok(())
}

fn require_tensor_shape(file: &GgufFile, name: &str, expected: &[usize]) -> Result<()> {
    let tensor = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "whisper-medusa: required tensor `{name}` is missing"
        ))
    })?;
    let shape = tensor
        .dimensions
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| {
                VokraError::ModelLoad(format!(
                    "whisper-medusa: tensor `{name}` dimension {dim} is not representable as usize"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if shape != expected {
        return Err(VokraError::ModelLoad(format!(
            "whisper-medusa: tensor `{name}` has shape {shape:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

/// Loaded official Whisper-Medusa v1 ASR engine.
pub struct WhisperMedusa {
    config: MedusaConfig,
    base: WhisperAsr,
    base_head: Arc<ResidualSiluLogitsAdapter>,
    weight_license: LicenseClass,
}

impl WhisperMedusa {
    /// Loads the strict metadata, all eleven head modules, and Whisper tower.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        match file
            .get(chunks::KEY_MODEL_ARCH)
            .and_then(GgufMetadataValue::as_str)
        {
            Some(ARCH) => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "whisper-medusa: GGUF arch `{other}`, expected `{ARCH}`"
                )));
            }
            None => return Err(missing_or_wrong(chunks::KEY_MODEL_ARCH, "a string")),
        }
        let config = MedusaConfig::from_gguf(file)?;
        validate_head_manifest(file, &config)?;

        let base = WhisperAsr::from_gguf(file)?;
        if base.model().config().d_model != config.hidden_size {
            return Err(VokraError::ModelLoad(format!(
                "whisper-medusa: base d_model {} != head hidden_size {}",
                base.model().config().d_model,
                config.hidden_size,
            )));
        }
        let weight = file.tensor_f32("medusa_heads.0.0.linear.weight")?;
        let bias = file.tensor_f32("medusa_heads.0.0.linear.bias")?;
        let base_head = Arc::new(ResidualSiluLogitsAdapter::new(
            weight,
            bias,
            config.hidden_size,
        )?);
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(GgufMetadataValue::as_str)
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        Ok(Self {
            config,
            base,
            base_head,
            weight_license,
        })
    }

    #[must_use]
    /// Attaches an explicit Whisper detokenizer.
    pub fn with_tokenizer(mut self, tokenizer: WhisperTokenizer) -> Self {
        self.base = self.base.with_tokenizer(tokenizer);
        self
    }

    #[must_use]
    /// Selects the compute backend. Module 0 uses the shared per-op Whisper
    /// path so its residual projection reaches the same backend as the base
    /// encoder/decoder; unsupported selections fail explicitly.
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.base = self.base.with_backend(backend);
        self
    }

    #[must_use]
    /// Returns the validated Medusa geometry.
    pub const fn config(&self) -> &MedusaConfig {
        &self.config
    }

    #[must_use]
    /// Number of speculative future-token heads (ten).
    pub const fn num_heads(&self) -> usize {
        self.config.num_heads
    }

    #[must_use]
    /// Total module count including the current-token module 0 (eleven).
    pub const fn module_count(&self) -> usize {
        self.config.module_count
    }

    #[must_use]
    /// Returns the artifact's stamped weight-license class.
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Official module-0 greedy tokens (not plain Whisper logits).
    pub fn transcribe_tokens(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        self.base
            .transcribe_tokens_with_output_adapter(pcm, Arc::clone(&self.base_head))
    }

    /// Module-0 vocabulary logits for the final token of an explicit decoder
    /// prefix.  Used by independent real-checkpoint parity and by callers that
    /// need to inspect the official base-head distribution without invoking
    /// the speculative tree driver.
    pub fn prefix_logits(&self, pcm: &[f32], prefix: &[u32]) -> Result<Vec<f32>> {
        self.base
            .prefix_logits_with_output_adapter(pcm, prefix, Arc::clone(&self.base_head))
    }

    /// Runs the official module-0 forward and detokenizes its greedy tokens.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        let tokens = self.transcribe_tokens(pcm)?;
        Ok(Transcription::new(self.base.render_ids(&tokens)?))
    }

    /// Accelerated Medusa tree decoding is a distinct API and never silently
    /// aliases the module-0 greedy path.
    pub fn transcribe_speculative(&self, _pcm: &[f32]) -> Result<Vec<u32>> {
        Err(VokraError::UnsupportedOp(format!(
            "whisper-medusa speculative tree decoding is not implemented: all {} future \
             heads are bound, but the draft/verify/accept tree-attention driver is absent; \
             use transcribe/transcribe_tokens for the official module-0 model forward",
            self.config.num_heads,
        )))
    }
}

impl AsrEngine for WhisperMedusa {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        Self::transcribe(self, pcm)
    }

    fn backend(&self) -> BackendKind {
        self.base.backend()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufArray, GgufBuilder};

    fn metadata_only() -> GgufFile {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        builder.add_string(KEY_REVISION, UPSTREAM_REVISION);
        builder.add_string(KEY_SOURCE_REVISION, UPSTREAM_SOURCE_REVISION);
        builder.add_string(KEY_CONFIG_SHA256, CONFIG_SHA256);
        builder.add_string(KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256);
        builder.add_u32(KEY_MEDUSA_NUM_HEADS, 10);
        builder.add_u32(KEY_MEDUSA_MODULE_COUNT, 11);
        builder.add_u32(KEY_MEDUSA_NUM_LAYERS, 1);
        builder.add_u32(KEY_MEDUSA_HIDDEN_SIZE, 1280);
        builder.add_string(KEY_MEDUSA_HEADS_TYPE, "base_head");
        builder.add_metadata(
            KEY_MEDUSA_CHOICES,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U32,
                values: EXPECTED_CHOICES
                    .iter()
                    .copied()
                    .map(GgufMetadataValue::U32)
                    .collect(),
            }),
        );
        builder.add_bool(KEY_MEDUSA_INIT_FROM_PROJ, true);
        builder.add_bool(KEY_OUTPUT_WHISPER_ORIGINAL, false);
        GgufFile::parse(builder.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn metadata_contract_reads_exact_official_values() {
        let file = metadata_only();
        let config = MedusaConfig::from_gguf(&file).unwrap();
        assert_eq!(config.num_heads, 10);
        assert_eq!(config.module_count, 11);
        assert_eq!(config.choices, EXPECTED_CHOICES);
    }

    #[test]
    fn missing_head_manifest_is_loud() {
        let file = metadata_only();
        let config = MedusaConfig::from_gguf(&file).unwrap();
        let error = validate_head_manifest(&file, &config).unwrap_err();
        assert!(error.to_string().contains("expected 22 head tensors"));
    }
}
