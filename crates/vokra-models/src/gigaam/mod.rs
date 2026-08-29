//! Strict boundary for Sber GigaAM v3 and Multilingual.
//!
//! GigaAM v3 is an RNNT release while GigaAM Multilingual is a CTC release.
//! Multilingual binds only the authenticated 552-tensor CTC contract; v3 stays
//! fail-closed because its RNNT route is not part of this implementation.

/// Authenticated Multilingual CTC binder and native CPU route.
pub mod multilingual;
/// Authenticated Multilingual CTC model binding and native CPU route.
pub use multilingual::GigaamMultilingual;

use vokra_core::backend::BackendKind;
use vokra_core::engines::AsrEngine;
use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::tasks::Transcription;
use vokra_core::{Result, VokraError};

/// v3 arch marker.
pub const ARCH_V3: &str = "sber_gigaam_v3";
/// multilingual arch marker.
pub const ARCH_MULTILINGUAL: &str = "gigaam_multilingual";
/// Arch markers recognized by this module.
pub const ACCEPTED_ARCHS: &[&str] = &[ARCH_V3, ARCH_MULTILINGUAL];
/// v3 model name.
pub const NAME_V3: &str = "gigaam-v3";
/// multilingual model name.
pub const NAME_MULTILINGUAL: &str = "sber-gigaam-multilingual";
/// Shared category marker.
pub const CATEGORY: &str = "asr";
/// v3 HF identity.
pub const UPSTREAM_HF_V3: &str = "ai-sage/GigaAM-v3";
/// multilingual source identity.
pub const UPSTREAM_URL_MULTILINGUAL: &str = "github.com/salute-developers/GigaAM";
/// Fixed weight/source license declaration.
pub const DEFAULT_LICENSE_SPDX: &str = "mit";
/// Primary source repository anchor.
pub const PRIMARY_SOURCE_REPO: &str = "github.com/salute-developers/GigaAM";
/// v3 HF source anchor.
pub const PRIMARY_SOURCE_HF_V3: &str = "huggingface.co/ai-sage/GigaAM-v3";
/// v3 converter path.
pub const CONVERTER_PATH_V3: &str = "crates/vokra-convert/src/models/sber_gigaam_v3.rs";
/// multilingual converter path.
pub const CONVERTER_PATH_MULTILINGUAL: &str =
    "crates/vokra-convert/src/models/sber_gigaam_multilingual.rs";
/// v3 preparation sidecar.
pub const SIDECAR_PATH_V3: &str = "tools/parity/sber_gigaam_v3_prepare_checkpoint.py";
/// multilingual preparation sidecar.
pub const SIDECAR_PATH_MULTILINGUAL: &str =
    "tools/parity/sber_gigaam_multilingual_prepare_checkpoint.py";
/// Model category metadata key.
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// v3 provenance metadata key.
pub const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
/// multilingual provenance metadata key.
pub const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";
/// Optional producer tensor declaration key retained for compatibility.
pub const KEY_REQUIRED_TENSORS: &str = "vokra.gigaam.required_tensors";
/// Historical topology marker retained for compatibility.
pub const LAYER_STACK_INFIX: &str = ".layers.";

/// Variant topology identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GigaamVariant {
    /// RNNT v3 (prediction and joint network).
    V3,
    /// Multilingual CTC (71 classes).
    Multilingual,
}

impl GigaamVariant {
    /// Parse a known arch marker for diagnostics.
    #[must_use]
    pub fn from_arch(arch: &str) -> Option<Self> {
        match arch {
            ARCH_V3 => Some(Self::V3),
            ARCH_MULTILINGUAL => Some(Self::Multilingual),
            _ => None,
        }
    }

    /// Arch marker.
    #[must_use]
    pub const fn arch(self) -> &'static str {
        match self {
            Self::V3 => ARCH_V3,
            Self::Multilingual => ARCH_MULTILINGUAL,
        }
    }

    /// Model name marker.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::V3 => NAME_V3,
            Self::Multilingual => NAME_MULTILINGUAL,
        }
    }

    /// CLI converter argument.
    #[must_use]
    pub const fn converter_arg(self) -> &'static str {
        match self {
            Self::V3 => "sber-gigaam-v3",
            Self::Multilingual => "sber-gigaam-multilingual",
        }
    }

    /// Primary topology distinction.
    #[must_use]
    pub const fn topology(self) -> &'static str {
        match self {
            Self::V3 => "RNNT (prediction + joint)",
            Self::Multilingual => "CTC (71 classes)",
        }
    }
}

/// Inspect an arch marker before selecting the variant binder.
pub fn verify_arch(file: &GgufFile) -> Result<GigaamVariant> {
    let arch = file
        .get(chunks::KEY_MODEL_ARCH)
        .and_then(|value| value.as_str())
        .ok_or_else(|| VokraError::ModelLoad("gigaam: missing model arch".to_owned()))?;
    GigaamVariant::from_arch(arch).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "gigaam: unsupported arch `{arch}`; expected one of {ACCEPTED_ARCHS:?}"
        ))
    })
}

/// Compatibility tensor surface. It is never populated by a successful
/// production bind in this inspection-only phase.
#[derive(Debug, Clone, Default)]
pub struct GigaamWeights {
    tensors: Vec<(String, Vec<usize>)>,
}

impl GigaamWeights {
    /// Refuse tensor binding until fixed manifests are authenticated.
    pub fn from_gguf(_gguf: &GgufFile) -> Result<Self> {
        Err(gigaam_inspection_error("tensor binding"))
    }

    /// Number of compatibility tensors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Compatibility accessor.
    #[must_use]
    pub fn tensors(&self) -> &[(String, Vec<usize>)] {
        &self.tensors
    }
}

/// Compatibility topology surface with no inferred geometry.
#[derive(Debug, Clone, Default)]
pub struct GigaamTopology;

impl GigaamTopology {
    /// Refuse topology inference from arbitrary tensors.
    pub fn probe(_weights: &GigaamWeights) -> Result<Self> {
        Err(gigaam_inspection_error("topology binding"))
    }

    /// No topology stacks are exposed.
    #[must_use]
    pub const fn stacks(&self) -> &[()] {
        &[]
    }
}

/// Runtime handle for the authenticated GigaAM family.
#[derive(Debug)]
pub enum Gigaam {
    /// Native Multilingual CTC route.
    Multilingual(GigaamMultilingual),
}

impl Gigaam {
    /// Bind the authenticated Multilingual CTC route. v3 remains fail-closed
    /// because its RNNT prediction/joint topology is outside this task.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let variant = verify_arch(file)?;
        match variant {
            GigaamVariant::V3 => Err(VokraError::UnsupportedOp(
                "gigaam: v3 RNNT native route is not implemented; refusing bind".to_owned(),
            )),
            GigaamVariant::Multilingual => {
                Ok(Self::Multilingual(GigaamMultilingual::from_gguf(file)?))
            }
        }
    }

    /// Filesystem loader using the same fail-closed binder.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// Variant accessor for a constructed compatibility handle.
    #[must_use]
    pub const fn variant(&self) -> GigaamVariant {
        match self {
            Self::Multilingual(_) => GigaamVariant::Multilingual,
        }
    }

    /// Run the selected native route.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<String> {
        match self {
            Self::Multilingual(model) => model.transcribe(pcm),
        }
    }

    /// Backend selected by the bound native route.
    pub fn backend(&self) -> BackendKind {
        match self {
            Self::Multilingual(model) => model.backend(),
        }
    }
}

impl AsrEngine for Gigaam {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        Ok(Transcription::new(self.transcribe(pcm)?))
    }

    fn backend(&self) -> BackendKind {
        self.backend()
    }
}

/// Loud fail-closed diagnostic for the two topology variants.
#[must_use]
pub fn transcribe_loud_partial(variant: GigaamVariant) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "gigaam: INSPECTION_ONLY; `{}` is {} but native forward, exact tensor manifest, frontend axes, and vocabulary are not authenticated; no transcript is emitted",
        variant.arch(),
        variant.topology()
    ))
}

fn gigaam_inspection_error(stage: &str) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "gigaam: INSPECTION_ONLY; refusing {stage} until fixed HF/source manifests and variant topology are authenticated"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_are_distinct_topologies() {
        assert_ne!(GigaamVariant::V3.arch(), GigaamVariant::Multilingual.arch());
        assert_ne!(
            GigaamVariant::V3.topology(),
            GigaamVariant::Multilingual.topology()
        );
        assert_eq!(GigaamVariant::from_arch(ARCH_V3), Some(GigaamVariant::V3));
        assert_eq!(
            GigaamVariant::from_arch(ARCH_MULTILINGUAL),
            Some(GigaamVariant::Multilingual)
        );
        assert_eq!(GigaamVariant::from_arch("sber_gigaam_ctc"), None);
    }

    #[test]
    fn v3_runtime_remains_fail_closed() {
        let error = VokraError::UnsupportedOp(
            "gigaam: v3 RNNT native route is not implemented; refusing bind".to_owned(),
        );
        assert!(matches!(error, VokraError::UnsupportedOp(_)));
    }
}
