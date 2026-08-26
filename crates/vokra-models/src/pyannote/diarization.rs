//! Exact Vokra-native execution of `pyannote/speaker-diarization-3.1`.
//!
//! [`PyannoteSpeakerDiarization31`] binds the weightless public pipeline
//! contract together with the exact PyanNet segmentation and WeSpeaker
//! embedding GGUFs. It reproduces the pinned upstream five-second sliding
//! windows, hard powerset conversion, overlap-aware masked embeddings,
//! cosine/centroid clustering, speaker counting and discrete reconstruction.
//!
//! # Primary source
//!
//! - Pipeline algorithm:
//!   <https://github.com/pyannote/pyannote-audio/blob/6a972c0c4e95de04637d7221208736c64c8b972a/pyannote/audio/pipelines/speaker_diarization.py>
//! - PyanNet backbone:
//!   `pyannote/audio/models/segmentation/PyanNet.py` at the same revision.
//! - The pipeline and PyanNet weights are MIT. The separately bound
//!   WeSpeaker checkpoint keeps its own CC-BY-4.0 attribution gate.
//!
//! No Python or pyannote runtime is linked. Unsupported backends and malformed
//! dependency artifacts fail explicitly; no operation silently falls back to
//! CPU.

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{CompliancePolicy, LicenseClass, Result, VokraError, check_weight_license};

mod pipeline31;

pub use pipeline31::{
    LOCAL_SPEAKERS, PyannoteSpeakerDiarization31, SAMPLE_RATE, SEGMENTATION_FRAMES,
    SEGMENTATION_STEP_SAMPLES, SEGMENTATION_WINDOW_SAMPLES,
};

/// Architecture tag of the public weightless pipeline GGUF.
pub const PIPELINE_MODEL_TAG: &str = "pyannote-speaker-diarization";
/// Callable runtime architecture tag.
pub const ARCH: &str = PIPELINE_MODEL_TAG;
/// Canonical model identity of the public weightless pipeline GGUF.
pub const PIPELINE_NAME: &str = "pyannote-speaker-diarization-3.1";
/// Semantic task category of the public pipeline GGUF.
pub const PIPELINE_CATEGORY: &str = "diarize";
/// Audited upstream pipeline repository.
pub const PIPELINE_UPSTREAM_HF: &str = "pyannote/speaker-diarization-3.1";
/// Exact pyannote.audio source revision used as the native pipeline oracle.
pub const PIPELINE_SOURCE_REVISION: &str = "6a972c0c4e95de04637d7221208736c64c8b972a";
/// Historical public Vokra pipeline repository.
pub const PIPELINE_PUBLIC_HF: &str = "vokra/pyannote-speaker-diarization-3.1";
/// Historical public Vokra pipeline revision.
pub const PIPELINE_PUBLIC_REVISION: &str = "a2bc759121b1cf64d3fc669be9785af963eb54b4";
/// Historical public pipeline filename.
pub const PIPELINE_PUBLIC_FILE: &str = "pyannote-speaker-diarization-3.1.gguf";
/// Historical public pipeline byte size.
pub const PIPELINE_PUBLIC_BYTES: u32 = 1_728;
/// Historical public pipeline SHA-256.
pub const PIPELINE_PUBLIC_SHA256: &str =
    "6f2fe6d681d75fdde84768792f54725baf4e5e025f3a9c4af9618867a64e3a64";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_PIPELINE_TYPE: &str = "vokra.pyannote_pipeline.type";
const KEY_PIPELINE_NAME: &str = "vokra.pyannote_pipeline.name";
const KEY_PIPELINE_VERSION: &str = "vokra.pyannote_pipeline.version";
const KEY_SEGMENTATION_MODEL: &str = "vokra.pyannote_pipeline.segmentation.model";
const KEY_SEGMENTATION_BATCH_SIZE: &str = "vokra.pyannote_pipeline.segmentation.batch_size";
const KEY_SEGMENTATION_MIN_DURATION_OFF: &str =
    "vokra.pyannote_pipeline.segmentation.min_duration_off";
const KEY_EMBEDDING_MODEL: &str = "vokra.pyannote_pipeline.embedding.model";
const KEY_EMBEDDING_BATCH_SIZE: &str = "vokra.pyannote_pipeline.embedding.batch_size";
const KEY_EMBEDDING_EXCLUDE_OVERLAP: &str = "vokra.pyannote_pipeline.embedding.exclude_overlap";
const KEY_CLUSTERING_ALGORITHM: &str = "vokra.pyannote_pipeline.clustering.algorithm";
const KEY_CLUSTERING_METHOD: &str = "vokra.pyannote_pipeline.clustering.method";
const KEY_CLUSTERING_MIN_CLUSTER_SIZE: &str = "vokra.pyannote_pipeline.clustering.min_cluster_size";
const KEY_CLUSTERING_THRESHOLD: &str = "vokra.pyannote_pipeline.clustering.threshold";

const EXPECTED_PIPELINE_TYPE: &str = "SpeakerDiarization";
const EXPECTED_PIPELINE_CLASS: &str = "pyannote.audio.pipelines.SpeakerDiarization";
const EXPECTED_PIPELINE_VERSION: &str = "3.1.0";
const EXPECTED_SEGMENTATION_MODEL: &str = "pyannote/segmentation-3.0";
const EXPECTED_EMBEDDING_MODEL: &str = "pyannote/wespeaker-voxceleb-resnet34-LM";
const EXPECTED_CLUSTERING_ALGORITHM: &str = "AgglomerativeClustering";
const EXPECTED_CLUSTERING_METHOD: &str = "centroid";
const EXPECTED_CLUSTERING_THRESHOLD: f32 = 0.704_565_5;

/// Strict typed view of the public `speaker-diarization-3.1` pipeline GGUF.
///
/// The upstream repository contains no learned tensors: this artifact is an
/// orchestration contract over the sibling segmentation and WeSpeaker GGUFs.
/// Loading rejects both missing metadata and any tensor payload.
#[derive(Debug, Clone, PartialEq)]
pub struct PyannoteSpeakerDiarization31Config {
    /// Upstream segmentation model identifier.
    pub segmentation_model: String,
    /// Upstream segmentation batch size (traceability; Vokra executes serially).
    pub segmentation_batch_size: u32,
    /// Gap filled by upstream final annotation post-processing, in seconds.
    pub segmentation_min_duration_off: f32,
    /// Upstream speaker-embedding model identifier.
    pub embedding_model: String,
    /// Upstream embedding batch size (traceability; Vokra executes serially).
    pub embedding_batch_size: u32,
    /// Whether overlapping frames are excluded from embedding masks when possible.
    pub embedding_exclude_overlap: bool,
    /// Minimum number of training embeddings retained as a large cluster.
    pub clustering_min_cluster_size: u32,
    /// Agglomerative-clustering cut height.
    pub clustering_threshold: f32,
}

impl PyannoteSpeakerDiarization31Config {
    /// Binds the exact public metadata-only pipeline contract.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_pipeline_string(file, chunks::KEY_MODEL_ARCH, PIPELINE_MODEL_TAG)?;
        require_pipeline_string(file, chunks::KEY_MODEL_NAME, PIPELINE_NAME)?;
        require_pipeline_string(file, KEY_MODEL_CATEGORY, PIPELINE_CATEGORY)?;
        require_pipeline_string(file, KEY_UPSTREAM_HF, PIPELINE_UPSTREAM_HF)?;
        require_pipeline_string(file, chunks::KEY_PROVENANCE_LICENSE, "mit")?;
        require_pipeline_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        )?;
        require_pipeline_string(file, chunks::KEY_PROVENANCE_MODEL_ID, PIPELINE_NAME)?;
        let license = check_weight_license(file, &CompliancePolicy::strict())?;
        if license.class != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "{PIPELINE_NAME}: weight license resolves to {}, expected permissive MIT",
                license.class.as_str()
            )));
        }
        if !file.tensors().is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "{PIPELINE_NAME}: pipeline GGUF must be weightless, found {} tensor(s)",
                file.tensors().len()
            )));
        }

        require_pipeline_string(file, KEY_PIPELINE_TYPE, EXPECTED_PIPELINE_TYPE)?;
        require_pipeline_string(file, KEY_PIPELINE_NAME, EXPECTED_PIPELINE_CLASS)?;
        require_pipeline_string(file, KEY_PIPELINE_VERSION, EXPECTED_PIPELINE_VERSION)?;
        require_pipeline_string(file, KEY_SEGMENTATION_MODEL, EXPECTED_SEGMENTATION_MODEL)?;
        require_pipeline_string(file, KEY_EMBEDDING_MODEL, EXPECTED_EMBEDDING_MODEL)?;
        require_pipeline_string(
            file,
            KEY_CLUSTERING_ALGORITHM,
            EXPECTED_CLUSTERING_ALGORITHM,
        )?;
        require_pipeline_string(file, KEY_CLUSTERING_METHOD, EXPECTED_CLUSTERING_METHOD)?;

        let segmentation_batch_size = required_pipeline_u32(file, KEY_SEGMENTATION_BATCH_SIZE)?;
        let segmentation_min_duration_off =
            required_pipeline_f32(file, KEY_SEGMENTATION_MIN_DURATION_OFF)?;
        let embedding_batch_size = required_pipeline_u32(file, KEY_EMBEDDING_BATCH_SIZE)?;
        let embedding_exclude_overlap =
            required_pipeline_bool(file, KEY_EMBEDDING_EXCLUDE_OVERLAP)?;
        let clustering_min_cluster_size =
            required_pipeline_u32(file, KEY_CLUSTERING_MIN_CLUSTER_SIZE)?;
        let clustering_threshold = required_pipeline_f32(file, KEY_CLUSTERING_THRESHOLD)?;

        for (key, actual, expected) in [
            (KEY_SEGMENTATION_BATCH_SIZE, segmentation_batch_size, 32),
            (KEY_EMBEDDING_BATCH_SIZE, embedding_batch_size, 32),
            (
                KEY_CLUSTERING_MIN_CLUSTER_SIZE,
                clustering_min_cluster_size,
                12,
            ),
        ] {
            if actual != expected {
                return Err(VokraError::ModelLoad(format!(
                    "{PIPELINE_NAME}: `{key}` is {actual}, expected {expected}"
                )));
            }
        }
        if segmentation_min_duration_off.to_bits() != 0.0f32.to_bits() {
            return Err(VokraError::ModelLoad(format!(
                "{PIPELINE_NAME}: `{KEY_SEGMENTATION_MIN_DURATION_OFF}` is {segmentation_min_duration_off}, expected 0"
            )));
        }
        if !embedding_exclude_overlap {
            return Err(VokraError::ModelLoad(format!(
                "{PIPELINE_NAME}: `{KEY_EMBEDDING_EXCLUDE_OVERLAP}` is false, expected true"
            )));
        }
        if clustering_threshold.to_bits() != EXPECTED_CLUSTERING_THRESHOLD.to_bits() {
            return Err(VokraError::ModelLoad(format!(
                "{PIPELINE_NAME}: `{KEY_CLUSTERING_THRESHOLD}` is {clustering_threshold}, expected {EXPECTED_CLUSTERING_THRESHOLD}"
            )));
        }

        Ok(Self {
            segmentation_model: EXPECTED_SEGMENTATION_MODEL.to_owned(),
            segmentation_batch_size,
            segmentation_min_duration_off,
            embedding_model: EXPECTED_EMBEDDING_MODEL.to_owned(),
            embedding_batch_size,
            embedding_exclude_overlap,
            clustering_min_cluster_size,
            clustering_threshold,
        })
    }
}

fn require_pipeline_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("{PIPELINE_NAME}: missing string `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{PIPELINE_NAME}: `{key}` is `{actual}`, expected `{expected}`"
        )));
    }
    Ok(())
}

fn required_pipeline_u32(file: &GgufFile, key: &str) -> Result<u32> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(value)) => Ok(*value),
        _ => Err(VokraError::ModelLoad(format!(
            "{PIPELINE_NAME}: missing u32 `{key}`"
        ))),
    }
}

fn required_pipeline_f32(file: &GgufFile, key: &str) -> Result<f32> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(value)) if value.is_finite() => Ok(*value),
        _ => Err(VokraError::ModelLoad(format!(
            "{PIPELINE_NAME}: missing finite f32 `{key}`"
        ))),
    }
}

fn required_pipeline_bool(file: &GgufFile, key: &str) -> Result<bool> {
    match file.get(key) {
        Some(GgufMetadataValue::Bool(value)) => Ok(*value),
        _ => Err(VokraError::ModelLoad(format!(
            "{PIPELINE_NAME}: missing bool `{key}`"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    fn pipeline_config_gguf(method: &str, with_tensor: bool) -> Vec<u8> {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, PIPELINE_MODEL_TAG);
        builder.add_string(chunks::KEY_MODEL_NAME, PIPELINE_NAME);
        builder.add_string(KEY_MODEL_CATEGORY, PIPELINE_CATEGORY);
        vokra_core::stamp_provenance(
            &mut builder,
            LicenseClass::Permissive,
            "mit",
            Some(PIPELINE_NAME),
            Some(PIPELINE_NAME),
            Some("pyannote/speaker-diarization-3.1 test pipeline"),
        );
        builder.add_string(KEY_UPSTREAM_HF, PIPELINE_UPSTREAM_HF);
        builder.add_string(KEY_PIPELINE_TYPE, EXPECTED_PIPELINE_TYPE);
        builder.add_string(KEY_PIPELINE_NAME, EXPECTED_PIPELINE_CLASS);
        builder.add_string(KEY_PIPELINE_VERSION, EXPECTED_PIPELINE_VERSION);
        builder.add_string(KEY_SEGMENTATION_MODEL, EXPECTED_SEGMENTATION_MODEL);
        builder.add_u32(KEY_SEGMENTATION_BATCH_SIZE, 32);
        builder.add_f32(KEY_SEGMENTATION_MIN_DURATION_OFF, 0.0);
        builder.add_string(KEY_EMBEDDING_MODEL, EXPECTED_EMBEDDING_MODEL);
        builder.add_u32(KEY_EMBEDDING_BATCH_SIZE, 32);
        builder.add_bool(KEY_EMBEDDING_EXCLUDE_OVERLAP, true);
        builder.add_string(KEY_CLUSTERING_ALGORITHM, EXPECTED_CLUSTERING_ALGORITHM);
        builder.add_string(KEY_CLUSTERING_METHOD, method);
        builder.add_u32(KEY_CLUSTERING_MIN_CLUSTER_SIZE, 12);
        builder.add_f32(KEY_CLUSTERING_THRESHOLD, EXPECTED_CLUSTERING_THRESHOLD);
        if with_tensor {
            builder
                .add_tensor(
                    "unexpected.weight",
                    GgmlType::F32,
                    vec![1],
                    0.0f32.to_le_bytes().to_vec(),
                )
                .expect("add unexpected tensor");
        }
        builder.to_bytes().expect("pipeline fixture GGUF")
    }

    #[test]
    fn strict_pipeline_config_binds_every_public_field() {
        let file = GgufFile::parse(pipeline_config_gguf(EXPECTED_CLUSTERING_METHOD, false))
            .expect("parse pipeline fixture");
        let config = PyannoteSpeakerDiarization31Config::from_gguf(&file)
            .expect("strict public pipeline contract");
        assert_eq!(config.segmentation_model, EXPECTED_SEGMENTATION_MODEL);
        assert_eq!(config.segmentation_batch_size, 32);
        assert_eq!(
            config.segmentation_min_duration_off.to_bits(),
            0.0f32.to_bits()
        );
        assert_eq!(config.embedding_model, EXPECTED_EMBEDDING_MODEL);
        assert_eq!(config.embedding_batch_size, 32);
        assert!(config.embedding_exclude_overlap);
        assert_eq!(config.clustering_min_cluster_size, 12);
        assert_eq!(
            config.clustering_threshold.to_bits(),
            EXPECTED_CLUSTERING_THRESHOLD.to_bits()
        );
    }

    #[test]
    fn strict_pipeline_config_rejects_wrong_clustering_method() {
        let file = GgufFile::parse(pipeline_config_gguf("average", false))
            .expect("parse pipeline fixture");
        let error = PyannoteSpeakerDiarization31Config::from_gguf(&file)
            .expect_err("speaker-diarization-3.1 requires centroid linkage");
        let message = error.to_string();
        assert!(message.contains(KEY_CLUSTERING_METHOD));
        assert!(message.contains("centroid"));
    }

    #[test]
    fn strict_pipeline_config_rejects_tensor_payload() {
        let file = GgufFile::parse(pipeline_config_gguf(EXPECTED_CLUSTERING_METHOD, true))
            .expect("parse pipeline fixture");
        let error = PyannoteSpeakerDiarization31Config::from_gguf(&file)
            .expect_err("pipeline config must remain weightless");
        assert!(error.to_string().contains("must be weightless"));
    }
}
