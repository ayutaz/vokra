//! NVIDIA Sortformer-Diar-4spk-v1 inspection-only runtime boundary.
//!
//! The native NEST FastConformer + Transformer + arrival-order diarization
//! implementation is not enabled. No GGUF metadata can authenticate itself;
//! therefore the public binder rejects every artifact until an owner-reviewed
//! VAST evidence bundle pins the exact upstream source, config and manifest.

use vokra_core::gguf::GgufFile;
use vokra_core::{LicenseClass, Result, VokraError};

/// Runtime architecture identifier.
pub const ARCH: &str = "sortformer";
/// Runtime model identifier.
pub const NAME: &str = "sortformer-diar-4spk-v1";
/// Required upstream weight license.
pub const LICENSE_SPDX: &str = "cc-by-nc-4.0";

/// Authenticated preparation marker reserved for a future reviewed artifact.
pub const GGUF_KEY_PREPARED_FORMAT: &str = "vokra.sortformer.prepared_format";
/// Reserved prepared-format value.
pub const PREPARED_FORMAT: &str = "vokra-sortformer-diar-4spk-v1-prepared-v1";

/// Source-reference Transformer `d_model = 192`; the NEST/Fast-Conformer
/// frontend has a distinct 512-dimensional encoder in the official config.
pub const DEFAULT_D_MODEL: u32 = 192;
/// Source-reference attention-head count.
pub const DEFAULT_N_HEADS: u32 = 8;
/// Source-reference NEST layer count.
pub const DEFAULT_NUM_NEST_LAYERS: u32 = 18;
/// Source-reference Transformer layer count.
pub const DEFAULT_NUM_TRANSFORMER_LAYERS: u32 = 18;
/// Source-reference speaker count.
pub const DEFAULT_NUM_SPEAKERS: u32 = 4;
/// Source-reference subsampling factor.
pub const DEFAULT_SUBSAMPLING_FACTOR: u32 = 8;

/// GGUF key for the source-reference d_model axis.
pub const GGUF_KEY_D_MODEL: &str = "vokra.sortformer.d_model";
/// GGUF key for the source-reference attention-head axis.
pub const GGUF_KEY_N_HEADS: &str = "vokra.sortformer.n_heads";
/// GGUF key for the source-reference NEST depth axis.
pub const GGUF_KEY_NUM_NEST_LAYERS: &str = "vokra.sortformer.num_nest_layers";
/// GGUF key for the source-reference Transformer depth axis.
pub const GGUF_KEY_NUM_TRANSFORMER_LAYERS: &str = "vokra.sortformer.num_transformer_layers";
/// GGUF key for the source-reference speaker-count axis.
pub const GGUF_KEY_NUM_SPEAKERS: &str = "vokra.sortformer.num_speakers";
/// GGUF key for the source-reference subsampling axis.
pub const GGUF_KEY_SUBSAMPLING_FACTOR: &str = "vokra.sortformer.subsampling_factor";
/// GGUF key for the authenticated frontend sample rate.
pub const GGUF_KEY_SAMPLE_RATE: &str = "vokra.sortformer.sample_rate";
/// GGUF key for the authenticated frontend mel-bin count.
pub const GGUF_KEY_N_MELS: &str = "vokra.sortformer.n_mels";
/// GGUF key for the authenticated upstream revision.
pub const GGUF_KEY_UPSTREAM_REVISION: &str = "vokra.sortformer.upstream_revision";
/// GGUF key for the authenticated source revision.
pub const GGUF_KEY_SOURCE_REVISION: &str = "vokra.sortformer.source_revision";
/// GGUF key for the authenticated checkpoint digest.
pub const GGUF_KEY_CHECKPOINT_SHA256: &str = "vokra.sortformer.checkpoint_sha256";
/// GGUF key for the authenticated config digest.
pub const GGUF_KEY_CONFIG_SHA256: &str = "vokra.sortformer.config_sha256";
/// GGUF key for the authenticated tensor-manifest digest.
pub const GGUF_KEY_TENSOR_MANIFEST_SHA256: &str = "vokra.sortformer.tensor_manifest_sha256";
/// GGUF key for the authenticated tensor count.
pub const GGUF_KEY_TENSOR_COUNT: &str = "vokra.sortformer.tensor_count";

/// Source-reference Sortformer axes. Defaults are never used by binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortformerConfig {
    /// Transformer post-encoder `d_model` (192; distinct from NEST width).
    pub d_model: u32,
    /// Multi-head attention head count.
    pub n_heads: u32,
    /// NEST encoder depth.
    pub num_nest_layers: u32,
    /// Transformer encoder depth.
    pub num_transformer_layers: u32,
    /// Maximum speaker count.
    pub num_speakers: u32,
    /// Encoder subsampling factor.
    pub subsampling_factor: u32,
}

impl Default for SortformerConfig {
    fn default() -> Self {
        Self::v1_default()
    }
}

impl SortformerConfig {
    /// Returns source-reference axes for tests and documentation only.
    #[must_use]
    pub const fn v1_default() -> Self {
        Self {
            d_model: DEFAULT_D_MODEL,
            n_heads: DEFAULT_N_HEADS,
            num_nest_layers: DEFAULT_NUM_NEST_LAYERS,
            num_transformer_layers: DEFAULT_NUM_TRANSFORMER_LAYERS,
            num_speakers: DEFAULT_NUM_SPEAKERS,
            subsampling_factor: DEFAULT_SUBSAMPLING_FACTOR,
        }
    }

    /// Reads explicitly stamped axes for offline inspection tooling.
    ///
    /// This low-level parser is not reachable through [`SortformerDiar`],
    /// whose public binder remains unconditionally disabled.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let required = |key: &str| {
            gguf.get(key)
                .and_then(|value| value.as_u64())
                .and_then(|value| u32::try_from(value).ok())
                .filter(|&value| value > 0)
                .ok_or_else(|| VokraError::ModelLoad(format!("sortformer: missing axis `{key}")))
        };
        Ok(Self {
            d_model: required(GGUF_KEY_D_MODEL)?,
            n_heads: required(GGUF_KEY_N_HEADS)?,
            num_nest_layers: required(GGUF_KEY_NUM_NEST_LAYERS)?,
            num_transformer_layers: required(GGUF_KEY_NUM_TRANSFORMER_LAYERS)?,
            num_speakers: required(GGUF_KEY_NUM_SPEAKERS)?,
            subsampling_factor: required(GGUF_KEY_SUBSAMPLING_FACTOR)?,
        })
    }
}

/// Low-level tensor inventory used only by future reviewed inspection code.
#[derive(Debug)]
pub struct SortformerWeights {
    tensors: Vec<(String, Vec<usize>)>,
}

impl SortformerWeights {
    /// Inventories GGUF tensors without making them a runnable model.
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
            return Err(VokraError::ModelLoad(
                "sortformer: zero tensors; inspection inventory is empty".to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Returns the number of inventoried tensors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Checks whether the first tensor dimension matches the source axis.
    #[must_use]
    pub fn matches_config(&self, config: &SortformerConfig) -> bool {
        self.tensors
            .first()
            .and_then(|(_, dims)| dims.first())
            .is_some_and(|&dim| dim == config.d_model as usize)
    }
}

/// A single diarization segment reserved for the future native forward path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerSegment {
    /// Speaker index in the model's arrival-order output.
    pub speaker_id: usize,
    /// Segment start in seconds.
    pub start_s: f32,
    /// Segment end in seconds.
    pub end_s: f32,
}

/// Sortformer runtime handle. Construction is disabled until evidence review.
#[derive(Debug)]
pub struct SortformerDiar {
    // Private seal: callers cannot construct a runtime handle around an
    // arbitrary GGUF and reach the diagnostic accessors.
    _sealed: (),
}

impl SortformerDiar {
    /// Refuses every GGUF until exact upstream evidence is reviewed and pinned.
    pub fn from_gguf(_file: &GgufFile) -> Result<Self> {
        Err(VokraError::ModelLoad(
            "sortformer: runtime binding is INSPECTION_ONLY and fail-closed until VAST evidence pins the exact HF checkpoint, NeMo source, config, license, and complete tensor manifest".to_owned(),
        ))
    }

    /// Returns source-reference axes; no instance can currently be created.
    #[must_use]
    pub const fn config(&self) -> &SortformerConfig {
        panic!("SortformerDiar is not constructible before evidence review")
    }

    /// Returns the weight license for a future reviewed instance.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        LicenseClass::Unknown
    }

    /// Returns zero because no runtime instance is constructible.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        0
    }

    /// Diarization remains explicitly unsupported in this inspection wave.
    pub fn diarize(&self, _pcm: &[f32]) -> Result<Vec<SpeakerSegment>> {
        Err(VokraError::UnsupportedOp(
            "sortformer diarize: native NEST FastConformer + Transformer + arrival-order head is not enabled".to_owned(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    #[test]
    fn public_binder_is_unconditionally_fail_closed() {
        let mut builder = GgufBuilder::new();
        builder.add_string("vokra.model.arch", ARCH);
        builder.add_string(GGUF_KEY_PREPARED_FORMAT, PREPARED_FORMAT);
        builder
            .add_tensor("arbitrary", GgmlType::F32, vec![1], vec![0; 4])
            .expect("tensor");
        let file = GgufFile::parse(builder.to_bytes().expect("GGUF")).expect("parse");
        let error = SortformerDiar::from_gguf(&file).unwrap_err().to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
        assert!(error.contains("fail-closed"), "{error}");
    }

    #[test]
    fn source_defaults_are_not_loader_fallbacks() {
        let builder = GgufBuilder::new();
        let file = GgufFile::parse(builder.to_bytes().expect("GGUF")).expect("parse");
        assert!(SortformerConfig::from_gguf(&file).is_err());
    }

    #[test]
    fn empty_low_level_inventory_is_rejected() {
        let builder = GgufBuilder::new();
        let file = GgufFile::parse(builder.to_bytes().expect("GGUF")).expect("parse");
        assert!(SortformerWeights::from_gguf(&file).is_err());
    }
}
