//! Native Suno Bark / Bark Small runtime contract (CPU / Metal).
//!
//! Bark is a three-stage hierarchical autoregressive model (semantic,
//! coarse-acoustic, and fine-acoustic Transformers) followed by the causal
//! EnCodec decoder embedded in the same checkpoint. This module accepts only
//! the two audited public Vokra manifests. The complete sorted tensor
//! name/shape digest is authenticated before any payload view is created, and
//! every canonical tensor must be F32 for the zero-copy mmap path.
//!
//! The historical public Full artifact carries the Small width/head metadata
//! (`768 / 12`) even though its authenticated 758-tensor manifest is the Full
//! `1024 / 16` topology. That one legacy stamp is accepted only after the
//! complete Full manifest has matched and is surfaced through
//! [`BarkModel::requires_metadata_repair`]. No count-only, partial, or
//! same-metadata substitute is accepted.

mod generation;
mod transformer;
mod weights;

pub use generation::{BarkGeneratedCodes, BarkGenerationConfig};

use std::path::Path;
use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

use self::weights::BarkMappedWeights;

/// GGUF architecture emitted by the offline converter.
pub const ARCH: &str = "bark";
/// Model-zoo category shared by both releases.
pub const CATEGORY: &str = "tts";
/// Embedded waveform sample rate.
pub const SAMPLE_RATE: u32 = 24_000;
/// EnCodec codebooks consumed by Bark's released synthesis route.
pub const CODEBOOKS_USED: usize = 8;
/// Entries in each EnCodec codebook.
pub const CODEBOOK_SIZE: usize = 1_024;
/// EnCodec latent width.
pub const CODEBOOK_DIM: usize = 128;

const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_VARIANT: &str = "vokra.bark.variant";
const KEY_MANIFEST_SHA256: &str = "vokra.bark.tensor_manifest_sha256";
const KEY_HIDDEN_SIZE: &str = "vokra.bark.hidden_size";
const KEY_NUM_HEADS: &str = "vokra.bark.num_heads";
const KEY_BLOCK_SIZE: &str = "vokra.bark.block_size";
const KEY_NUM_LAYERS: &str = "vokra.bark.num_layers_per_stage";
const KEY_SEMANTIC_INPUT_VOCAB: &str = "vokra.bark.semantic.input_vocab_size";
const KEY_SEMANTIC_OUTPUT_VOCAB: &str = "vokra.bark.semantic.output_vocab_size";
const KEY_COARSE_INPUT_VOCAB: &str = "vokra.bark.coarse.input_vocab_size";
const KEY_COARSE_OUTPUT_VOCAB: &str = "vokra.bark.coarse.output_vocab_size";
const KEY_FINE_INPUT_VOCAB: &str = "vokra.bark.fine.input_vocab_size";
const KEY_FINE_OUTPUT_VOCAB: &str = "vokra.bark.fine.output_vocab_size";
const KEY_FINE_N_CODES_TOTAL: &str = "vokra.bark.fine.n_codes_total";
const KEY_FINE_N_CODES_GIVEN: &str = "vokra.bark.fine.n_codes_given";
const KEY_CODEC_UPSTREAM_HF: &str = "vokra.bark.codec.upstream_hf";
const KEY_CODEC_SAMPLE_RATE: &str = "vokra.bark.codec.sample_rate";

/// Learned operations required by the released three-stage LM plus embedded
/// EnCodec waveform decoder. A selected backend must cover this complete set
/// before generation starts; there is no implicit CPU substitution.
pub const BARK_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Elu,
    HotOp::Tanh,
    HotOp::Conv1d,
    HotOp::EncodecRvq,
];

/// Authenticated public Bark release.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarkVariant {
    /// `suno/bark-small`: 768 hidden, 12 heads, 12 layers per LM stage.
    Small,
    /// `suno/bark`: 1024 hidden, 16 heads, 24 layers per LM stage.
    Full,
}

impl BarkVariant {
    /// Canonical Vokra model name.
    pub const fn model_name(self) -> &'static str {
        match self {
            Self::Small => "bark-small",
            Self::Full => "bark",
        }
    }

    /// Short GGUF variant tag.
    pub const fn variant_tag(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Full => "full",
        }
    }

    /// Official upstream Hugging Face repository.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Small => "suno/bark-small",
            Self::Full => "suno/bark",
        }
    }

    /// Immutable upstream checkpoint revision.
    pub const fn upstream_revision(self) -> &'static str {
        match self {
            Self::Small => "1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd",
            Self::Full => "70a8a7d34168586dc5d028fa9666aceade177992",
        }
    }

    /// Audited Vokra Hugging Face revision containing the public GGUF.
    pub const fn public_gguf_revision(self) -> &'static str {
        match self {
            Self::Small => "09802c56a2b2e8ad87835115b94b38031fde29b6",
            Self::Full => "f304ddcdfd9218994731ec3b09e89b9961b8b751",
        }
    }

    /// SHA-256 of the complete public GGUF file.
    pub const fn public_gguf_sha256(self) -> &'static str {
        match self {
            Self::Small => "43b781a0dcd66f1e7451005e461ec20e2141bc9c4f529feb4a9a8c0e352ea137",
            Self::Full => "fd628312ce7d8e1cbc41718741614116d5c7f08d0763f81622edbac320b208ec",
        }
    }

    /// Transformer width shared by all three LM stages.
    pub const fn hidden_size(self) -> usize {
        match self {
            Self::Small => 768,
            Self::Full => 1_024,
        }
    }

    /// Self-attention head count shared by all three LM stages.
    pub const fn num_heads(self) -> usize {
        match self {
            Self::Small => 12,
            Self::Full => 16,
        }
    }

    /// Transformer layer count in each LM stage.
    pub const fn num_layers(self) -> usize {
        match self {
            Self::Small => 12,
            Self::Full => 24,
        }
    }

    /// Complete tensor count including the embedded codec.
    pub const fn tensor_count(self) -> usize {
        match self {
            Self::Small => 518,
            Self::Full => 758,
        }
    }

    /// Canonical complete tensor-manifest SHA-256 as lowercase hex.
    pub const fn tensor_manifest_sha256(self) -> &'static str {
        match self {
            Self::Small => "25adef111ab1318346c4f54003bdfa7dc3305bc1b20fdcbd3a9cdfbe1e4ff127",
            Self::Full => "c32d8b203779ea68235c0304152781315a8a18694938c4872bfe476ea0da6424",
        }
    }

    const fn from_tensor_count(count: usize) -> Option<Self> {
        match count {
            518 => Some(Self::Small),
            758 => Some(Self::Full),
            _ => None,
        }
    }

    const fn spec(self) -> StrictCheckpointSpec {
        match self {
            Self::Small => StrictCheckpointSpec {
                label: "bark/small",
                arch: ARCH,
                model_name: "bark-small",
                model_name_alias: None,
                tensor_count: 518,
                manifest_sha256: [
                    0x25, 0xad, 0xef, 0x11, 0x1a, 0xb1, 0x31, 0x83, 0x46, 0xc4, 0xf5, 0x40, 0x03,
                    0xbd, 0xfa, 0x7d, 0xc3, 0x30, 0x5b, 0xc1, 0xb2, 0x0f, 0xdc, 0xbd, 0x3a, 0x9c,
                    0xdf, 0xbe, 0x1e, 0x4f, 0xf1, 0x27,
                ],
            },
            Self::Full => StrictCheckpointSpec {
                label: "bark/full",
                arch: ARCH,
                model_name: "bark",
                model_name_alias: None,
                tensor_count: 758,
                manifest_sha256: [
                    0xc3, 0x2d, 0x8b, 0x20, 0x37, 0x79, 0xea, 0x68, 0x23, 0x5c, 0x03, 0x04, 0x15,
                    0x27, 0x81, 0x31, 0x5a, 0x8a, 0x18, 0x69, 0x49, 0x38, 0xc4, 0x87, 0x2b, 0xfe,
                    0x47, 0x6e, 0xa0, 0xda, 0x64, 0x24,
                ],
            },
        }
    }
}

/// Immutable topology selected by the authenticated release manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BarkConfig {
    /// Full or Small release.
    pub variant: BarkVariant,
    /// Transformer hidden width.
    pub hidden_size: usize,
    /// Attention head count.
    pub num_heads: usize,
    /// Layers in each of the three Transformer stages.
    pub num_layers_per_stage: usize,
    /// Maximum sequence positions.
    pub block_size: usize,
}

impl BarkConfig {
    const fn for_variant(variant: BarkVariant) -> Self {
        Self {
            variant,
            hidden_size: variant.hidden_size(),
            num_heads: variant.num_heads(),
            num_layers_per_stage: variant.num_layers(),
            block_size: 1_024,
        }
    }
}

/// Strictly authenticated Bark model, optionally owning its mmap payload.
#[derive(Debug, Clone)]
pub struct BarkModel {
    config: BarkConfig,
    backend: BackendKind,
    weight_license: LicenseClass,
    requires_metadata_repair: bool,
    mapped: Option<BarkMappedWeights>,
}

impl BarkModel {
    /// Opens a public Bark GGUF through the true mmap path on CPU.
    pub fn open_mapped(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_mapped_with_backend(path, BackendKind::Cpu)
    }

    /// Opens a public Bark GGUF through mmap and preflights one complete
    /// execution backend before binding payload views.
    pub fn open_mapped_with_backend(path: impl AsRef<Path>, backend: BackendKind) -> Result<Self> {
        let file = vokra_mmap::open_gguf(path.as_ref()).map_err(VokraError::from)?;
        Self::from_gguf_mapped_with_backend(Arc::new(file), backend)
    }

    /// Descriptor-only authentication. Use [`Self::open_mapped`] or
    /// [`Self::from_gguf_mapped`] for executable inference.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let variant = BarkVariant::from_tensor_count(file.tensors().len()).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "bark: tensor count {} matches neither Bark Small (518) nor Bark Full (758)",
                file.tensors().len()
            ))
        })?;
        let checkpoint = StrictCheckpoint::bind(file, variant.spec())?;
        let repair = validate_metadata(file, variant)?;
        validate_f32_descriptors(file, variant)?;
        if checkpoint.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "bark/{}: pinned Suno MIT checkpoint must classify as permissive, got {:?}",
                variant.variant_tag(),
                checkpoint.weight_license()
            )));
        }
        debug_assert_eq!(checkpoint.tensor_count(), variant.tensor_count());
        Ok(Self {
            config: BarkConfig::for_variant(variant),
            backend: BackendKind::Cpu,
            weight_license: checkpoint.weight_license(),
            requires_metadata_repair: repair,
            mapped: None,
        })
    }

    /// Descriptor-only authentication plus whole-model backend preflight.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        let mut model = Self::from_gguf(file)?;
        let _ = Compute::for_backend(backend, BARK_HOT_OPS)?;
        model.backend = backend;
        Ok(model)
    }

    /// Strictly authenticates and retains an already mmap-backed artifact.
    pub fn from_gguf_mapped(file: Arc<GgufFile>) -> Result<Self> {
        Self::from_gguf_mapped_with_backend(file, BackendKind::Cpu)
    }

    /// Strictly authenticates and retains an mmap-backed artifact for one
    /// explicit CPU or Metal backend. All F32 views are alignment-checked;
    /// tensor payloads are not copied into a second resident allocation.
    pub fn from_gguf_mapped_with_backend(
        file: Arc<GgufFile>,
        backend: BackendKind,
    ) -> Result<Self> {
        let mut model = Self::from_gguf(&file)?;
        let _ = Compute::for_backend(backend, BARK_HOT_OPS)?;
        model.mapped = Some(BarkMappedWeights::bind(file, &model.config)?);
        model.backend = backend;
        Ok(model)
    }

    /// Authenticated release.
    pub const fn variant(&self) -> BarkVariant {
        self.config.variant
    }

    /// Exact topology selected from the manifest.
    pub const fn config(&self) -> &BarkConfig {
        &self.config
    }

    /// Selected whole-model backend.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Fail-closed weight-license classification.
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Whether the accepted public artifact needs canonical additive metadata
    /// restamping. This never weakens manifest authentication.
    pub const fn requires_metadata_repair(&self) -> bool {
        self.requires_metadata_repair
    }

    /// Whether this value owns a true mapped payload usable by inference.
    pub const fn is_mapped(&self) -> bool {
        self.mapped.is_some()
    }

    pub(super) fn mapped(&self) -> Result<&BarkMappedWeights> {
        self.mapped.as_ref().ok_or_else(|| {
            VokraError::ModelLoad(
                "bark: executable inference requires open_mapped or from_gguf_mapped; the borrowed from_gguf constructor authenticates descriptors only"
                    .to_owned(),
            )
        })
    }
}

fn validate_metadata(file: &GgufFile, variant: BarkVariant) -> Result<bool> {
    let label = format!("bark/{}", variant.variant_tag());
    require_string(file, "vokra.model.category", CATEGORY, &label)?;
    require_string(file, KEY_VARIANT, variant.variant_tag(), &label)?;
    require_string(file, KEY_UPSTREAM_HF, variant.upstream_hf(), &label)?;
    require_string(
        file,
        chunks::KEY_PROVENANCE_MODEL_ID,
        variant.model_name(),
        &label,
    )?;
    require_string(
        file,
        chunks::KEY_PROVENANCE_SOURCE,
        variant.upstream_hf(),
        &label,
    )?;
    require_string(file, chunks::KEY_PROVENANCE_LICENSE, "mit", &label)?;
    require_string(
        file,
        chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
        LicenseClass::Permissive.as_str(),
        &label,
    )?;

    let hidden = required_u64(file, KEY_HIDDEN_SIZE, &label)? as usize;
    let heads = required_u64(file, KEY_NUM_HEADS, &label)? as usize;
    let legacy_full_axes = variant == BarkVariant::Full && hidden == 768 && heads == 12;
    if !legacy_full_axes && (hidden != variant.hidden_size() || heads != variant.num_heads()) {
        return Err(VokraError::ModelLoad(format!(
            "{label}: hidden/head metadata is {hidden}/{heads}; expected {}/{}{}",
            variant.hidden_size(),
            variant.num_heads(),
            if variant == BarkVariant::Full {
                " (the historical 768/12 stamp is accepted only behind the exact Full manifest)"
            } else {
                ""
            }
        )));
    }

    for (key, expected) in [
        (KEY_BLOCK_SIZE, 1_024usize),
        (KEY_NUM_LAYERS, variant.num_layers()),
        (KEY_SEMANTIC_INPUT_VOCAB, 129_600),
        (KEY_SEMANTIC_OUTPUT_VOCAB, 10_048),
        (KEY_COARSE_INPUT_VOCAB, 12_096),
        (KEY_COARSE_OUTPUT_VOCAB, 12_096),
        (KEY_FINE_INPUT_VOCAB, 1_056),
        (KEY_FINE_OUTPUT_VOCAB, 1_056),
        (KEY_FINE_N_CODES_TOTAL, 8),
        (KEY_FINE_N_CODES_GIVEN, 1),
        (KEY_CODEC_SAMPLE_RATE, SAMPLE_RATE as usize),
    ] {
        require_u64(file, key, expected as u64, &label)?;
    }
    require_string(
        file,
        KEY_CODEC_UPSTREAM_HF,
        "facebook/encodec_24khz",
        &label,
    )?;

    let revision_missing = validate_optional_string(
        file,
        KEY_UPSTREAM_REVISION,
        variant.upstream_revision(),
        &label,
    )?;
    let manifest_missing = validate_optional_string(
        file,
        KEY_MANIFEST_SHA256,
        variant.tensor_manifest_sha256(),
        &label,
    )?;
    Ok(legacy_full_axes || revision_missing || manifest_missing)
}

fn validate_f32_descriptors(file: &GgufFile, variant: BarkVariant) -> Result<()> {
    for tensor in file.tensors() {
        if tensor.dtype != vokra_core::gguf::GgmlType::F32 {
            return Err(VokraError::ModelLoad(format!(
                "bark/{}: tensor `{}` is {:?}; the canonical mmap contract is entirely F32",
                variant.variant_tag(),
                tensor.name,
                tensor.dtype
            )));
        }
    }
    Ok(())
}

fn required_string<'a>(file: &'a GgufFile, key: &str, label: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("{label}: missing/non-string `{key}`")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str, label: &str) -> Result<()> {
    let actual = required_string(file, key, label)?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{label}: `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn required_u64(file: &GgufFile, key: &str, label: &str) -> Result<u64> {
    file.get(key)
        .and_then(GgufMetadataValue::as_u64)
        .ok_or_else(|| VokraError::ModelLoad(format!("{label}: missing/non-integer `{key}`")))
}

fn require_u64(file: &GgufFile, key: &str, expected: u64, label: &str) -> Result<()> {
    let actual = required_u64(file, key, label)?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{label}: `{key}`={actual}, expected {expected}"
        )));
    }
    Ok(())
}

/// Returns `true` when the additive key is absent and therefore needs a
/// provenance-only restamp. A present wrong value always fails closed.
fn validate_optional_string(
    file: &GgufFile,
    key: &str,
    expected: &str,
    label: &str,
) -> Result<bool> {
    match file.get(key) {
        None => Ok(true),
        Some(value) => {
            let actual = value.as_str().ok_or_else(|| {
                VokraError::ModelLoad(format!("{label}: `{key}` is present but non-string"))
            })?;
            if actual != expected {
                return Err(VokraError::ModelLoad(format!(
                    "{label}: `{key}`={actual:?}, expected {expected:?}"
                )));
            }
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_axes_are_manifest_selected() {
        assert_eq!(
            BarkVariant::from_tensor_count(518),
            Some(BarkVariant::Small)
        );
        assert_eq!(BarkVariant::from_tensor_count(758), Some(BarkVariant::Full));
        assert_eq!(BarkVariant::from_tensor_count(517), None);
        assert_eq!(BarkVariant::from_tensor_count(759), None);
        assert_eq!(BarkConfig::for_variant(BarkVariant::Small).hidden_size, 768);
        assert_eq!(
            BarkConfig::for_variant(BarkVariant::Full).hidden_size,
            1_024
        );
        assert_eq!(BarkVariant::Full.num_heads(), 16);
        assert_eq!(BarkVariant::Full.num_layers(), 24);
    }

    #[test]
    fn public_identity_pins_are_immutable() {
        assert_eq!(BarkVariant::Small.tensor_count(), 518);
        assert_eq!(BarkVariant::Full.tensor_count(), 758);
        assert_eq!(BarkVariant::Small.tensor_manifest_sha256().len(), 64);
        assert_eq!(BarkVariant::Full.tensor_manifest_sha256().len(), 64);
        assert_ne!(
            BarkVariant::Small.public_gguf_sha256(),
            BarkVariant::Full.public_gguf_sha256()
        );
        assert_ne!(
            BarkVariant::Small.upstream_revision(),
            BarkVariant::Full.upstream_revision()
        );
    }
}
