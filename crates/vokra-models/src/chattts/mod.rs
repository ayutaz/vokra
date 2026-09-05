//! ChatTTS (`2Noise/ChatTTS`, CC-BY-NC-4.0) inspection-only runtime boundary.
//!
//! The upstream release is a composite GPT + Embed + DVAE + Decoder + Vocos
//! bundle. The historical public artifact contains only GPT. Until composite
//! binding, a clean-room native forward/parity result, and legal/publication
//! gates are complete, every public load route fails closed. No upstream
//! implementation is copied or executed by this runtime.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{CompliancePolicy, LicenseClass, Result, VokraError};

mod components;
mod gpt;

pub use components::{
    ChatTtsComponentContract, ChatTtsCompositeSession, ChatTtsDecoderConfig, ChatTtsDvaeConfig,
    ChatTtsGfsqConfig, ChatTtsVocosConfig,
};
pub use gpt::{
    AUDIO_CODEBOOKS, AUDIO_VOCAB_SIZE, ChatTtsGptConfig, ChatTtsGptSession, ChatTtsPrompt,
    ChatTtsSamplingConfig, EOS_AUDIO_CODE, HIDDEN_SIZE, INTERMEDIATE_SIZE, MAX_POSITION_EMBEDDINGS,
    NUM_HEADS, NUM_LAYERS, TEXT_VOCAB_SIZE,
};

/// GGUF architecture identifier for ChatTTS.
pub const ARCH: &str = "chattts";
/// Stable model name used by the inspection boundary.
pub const NAME: &str = "chattts";
/// Vokra model category for ChatTTS.
pub const CATEGORY: &str = "tts";
/// Upstream Hugging Face repository identifier.
pub const UPSTREAM_HF: &str = "2Noise/ChatTTS";
/// Default SPDX license identifier for the upstream weights.
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-4.0";
/// GGUF metadata key for the Vokra model category.
pub const GGUF_KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// GGUF metadata key for the upstream Hugging Face provenance.
pub const GGUF_KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Human-readable upstream weights location.
pub const PRIMARY_SOURCE_UPSTREAM_HF: &str = "huggingface.co/2Noise/ChatTTS";
/// Human-readable upstream source-code location.
pub const PRIMARY_SOURCE_CODE: &str = "github.com/2noise/ChatTTS";
/// Repository ticket tracking ChatTTS coverage.
pub const PRIMARY_SOURCE_TICKET: &str = "docs/tickets/coverage-audit-2026-08-03/wave-d/chattts.md";

/// Legacy inspection label retained for callers that display old metadata.
#[deprecated(
    note = "ChatTTS uses the VAST inspection workflow; this is not a runtime prerequisite"
)]
pub const PREP_SCRIPT_PATH: &str = "tools/parity/chattts_inspect.py";
/// Legacy metadata namespace label; no runtime axis group is currently read.
#[deprecated(note = "ChatTTS is inspection-only and has no runtime axis contract")]
pub const AXIS_GROUP_PREFIX: &str = "vokra.chattts.";

/// Tensor-name prefix for the GPT module.
pub const MODULE_PREFIX_GPT: &str = "gpt.";
/// Tensor-name prefix for the DVAE module.
pub const MODULE_PREFIX_DVAE: &str = "dvae.";
/// Tensor-name prefix for the Vocos module.
pub const MODULE_PREFIX_VOCOS: &str = "vocos.";
/// Tensor-name prefix for speaker statistics tensors.
pub const MODULE_PREFIX_SPEAKER_STATS: &str = "spk_stat";
/// Display label for the GPT module.
pub const MODULE_LABEL_GPT: &str = "GPT autoregressive backbone";
/// Display label for the DVAE module.
pub const MODULE_LABEL_DVAE: &str = "DVAE speech-token decoder";
/// Display label for the Vocos module.
pub const MODULE_LABEL_VOCOS: &str = "Vocos vocoder head";

/// Counts recognized ChatTTS module tensors in an inspected GGUF manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatTtsModuleCensus {
    /// Number of tensors with the GPT prefix.
    pub gpt: usize,
    /// Number of tensors with the DVAE prefix.
    pub dvae: usize,
    /// Number of tensors with the Vocos prefix.
    pub vocos: usize,
    /// Number of speaker-statistics tensors.
    pub speaker_stats: usize,
    /// Total number of tensors in the inspected GGUF.
    pub total_tensors: usize,
}

impl ChatTtsModuleCensus {
    /// Whether any recognized ChatTTS module tensor was found.
    #[must_use]
    pub const fn matched_any(&self) -> bool {
        self.gpt > 0 || self.dvae > 0 || self.vocos > 0 || self.speaker_stats > 0
    }

    /// Whether GPT, DVAE, and Vocos tensors are all present.
    #[must_use]
    pub const fn synthesis_chain_complete(&self) -> bool {
        self.gpt > 0 && self.dvae > 0 && self.vocos > 0
    }

    /// Return display labels for synthesis modules that are absent.
    #[must_use]
    pub fn missing_synthesis_modules(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if self.gpt == 0 {
            missing.push(MODULE_LABEL_GPT);
        }
        if self.dvae == 0 {
            missing.push(MODULE_LABEL_DVAE);
        }
        if self.vocos == 0 {
            missing.push(MODULE_LABEL_VOCOS);
        }
        missing
    }
}

/// Legacy GGUF manifest introspection. This type never implies runtime support.
#[derive(Debug, Clone)]
pub struct ChatTtsWeights {
    tensors: Vec<(String, Vec<usize>)>,
}

impl ChatTtsWeights {
    /// Inspect tensor names and dimensions from a GGUF manifest.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let tensors = gguf
            .tensors()
            .iter()
            .map(|info| {
                (
                    info.name.clone(),
                    info.dimensions.iter().map(|&dim| dim as usize).collect(),
                )
            })
            .collect::<Vec<_>>();
        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "chattts: zero-tensor GGUF is invalid legacy manifest evidence; no runtime bind is available (source: {PRIMARY_SOURCE_UPSTREAM_HF})"
            )));
        }
        Ok(Self { tensors })
    }

    /// Return the number of tensors in the inspected manifest.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Return all inspected tensor names.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// Return dimensions for a named tensor, if present.
    #[must_use]
    pub fn tensor_dims(&self, name: &str) -> Option<&[usize]> {
        self.tensors
            .iter()
            .find(|(tensor, _)| tensor == name)
            .map(|(_, dims)| dims.as_slice())
    }

    /// Count tensors whose names start with `prefix`.
    #[must_use]
    pub fn count_with_prefix(&self, prefix: &str) -> usize {
        self.tensors
            .iter()
            .filter(|(name, _)| name.starts_with(prefix))
            .count()
    }

    /// Summarize recognized module prefixes in the manifest.
    #[must_use]
    pub fn module_census(&self) -> ChatTtsModuleCensus {
        ChatTtsModuleCensus {
            gpt: self.count_with_prefix(MODULE_PREFIX_GPT),
            dvae: self.count_with_prefix(MODULE_PREFIX_DVAE),
            vocos: self.count_with_prefix(MODULE_PREFIX_VOCOS),
            speaker_stats: self.count_with_prefix(MODULE_PREFIX_SPEAKER_STATS),
            total_tensors: self.tensors.len(),
        }
    }

    /// Require a named tensor to be present in the manifest.
    pub fn require_tensor(&self, name: &str) -> Result<&[usize]> {
        self.tensor_dims(name).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "chattts: legacy manifest tensor `{name}` is absent; manifest introspection cannot substitute tensors"
            ))
        })
    }

    /// Require a named tensor to have exactly `expected` dimensions.
    pub fn require_tensor_dims(&self, name: &str, expected: &[usize]) -> Result<()> {
        let actual = self.require_tensor(name)?;
        if actual != expected {
            return Err(VokraError::ModelLoad(format!(
                "chattts: legacy tensor `{name}` has dims {actual:?}, expected {expected:?}"
            )));
        }
        Ok(())
    }
}

/// Compatibility handle whose public load routes are unconditionally refused.
#[derive(Debug, Clone)]
pub struct ChatTts {
    name: Option<String>,
    category: Option<String>,
    upstream_hf: Option<String>,
    model_id: Option<String>,
    source: Option<String>,
    weights: ChatTtsWeights,
    weight_license: LicenseClass,
    attribution: Option<String>,
}

impl ChatTts {
    /// Refuses correctly tagged artifacts until composite native support exists.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        verify_arch(file)?;
        Err(VokraError::UnsupportedOp(
            "chattts: INSPECTION_ONLY — composite binding, clean-room native forward/parity, and legal/publication gates are incomplete; no GGUF is runtime-ready".to_owned(),
        ))
    }

    /// A policy cannot bypass the inspection-only runtime boundary.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|error| VokraError::ModelLoad(format!("chattts GGUF: {error}")))?;
        verify_arch(&file)?;
        let _ = policy;
        Err(VokraError::UnsupportedOp(
            "chattts: INSPECTION_ONLY — compliance policy cannot enable an unauthenticated composite runtime".to_owned(),
        ))
    }

    /// Refuse loading from a path through the strict inspection policy.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_path_with_policy(path, &CompliancePolicy::strict())
    }

    /// Refuse loading from a path with an explicit compliance policy.
    pub fn from_path_with_policy(
        path: impl AsRef<std::path::Path>,
        policy: &CompliancePolicy,
    ) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(VokraError::Io)?;
        Self::from_gguf_with_policy(&bytes, policy)
    }

    /// Return the optional model name metadata.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    /// Return the optional model category metadata.
    #[must_use]
    pub fn category(&self) -> Option<&str> {
        self.category.as_deref()
    }
    /// Return the optional upstream Hugging Face identifier.
    #[must_use]
    pub fn upstream_hf(&self) -> Option<&str> {
        self.upstream_hf.as_deref()
    }
    /// Return the optional model identifier.
    #[must_use]
    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }
    /// Return the optional source metadata.
    #[must_use]
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }
    /// Return optional attribution metadata.
    #[must_use]
    pub fn attribution(&self) -> Option<&str> {
        self.attribution.as_deref()
    }
    /// Return the inspected GGUF manifest.
    #[must_use]
    pub fn weights(&self) -> &ChatTtsWeights {
        &self.weights
    }
    /// Return the recognized module census.
    #[must_use]
    pub fn module_census(&self) -> ChatTtsModuleCensus {
        self.weights.module_census()
    }
    /// Return the recorded weight license classification.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Compatibility API; a handle cannot be constructed through public loads.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        let _ = text;
        Err(VokraError::UnsupportedOp(
            "chattts: INSPECTION_ONLY — native composite forward/parity is not available; no waveform is fabricated".to_owned(),
        ))
    }
}

fn verify_arch(file: &GgufFile) -> Result<()> {
    match file
        .get(chunks::KEY_MODEL_ARCH)
        .and_then(|value| value.as_str())
    {
        Some(actual) if actual == ARCH => Ok(()),
        Some(actual) => Err(VokraError::ModelLoad(format!(
            "chattts: GGUF arch `{actual}` does not match expected `{ARCH}`"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "chattts: GGUF is missing vokra.model.arch; expected `{ARCH}`"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    #[test]
    fn public_load_routes_are_fail_closed() {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        builder
            .add_tensor("gpt.probe", GgmlType::F32, vec![1, 1], vec![0; 4])
            .expect("tensor");
        let bytes = builder.to_bytes().expect("GGUF");
        let file = GgufFile::parse(bytes.clone()).expect("parse");
        assert!(
            matches!(ChatTts::from_gguf(&file), Err(VokraError::UnsupportedOp(message)) if message.contains("INSPECTION_ONLY"))
        );
        assert!(
            matches!(ChatTts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict()), Err(VokraError::UnsupportedOp(message)) if message.contains("INSPECTION_ONLY"))
        );
    }

    #[test]
    fn legacy_gpt_only_manifest_cannot_bind() {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        builder
            .add_tensor("gpt.layer", GgmlType::F32, vec![1, 1], vec![0; 4])
            .expect("tensor");
        let file = GgufFile::parse(builder.to_bytes().expect("GGUF")).expect("parse");
        assert!(ChatTts::from_gguf(&file).is_err());
    }
}
