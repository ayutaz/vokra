//! Native deepfake-audio classification on Mac CPU and Metal.
//!
//! The canonical checkpoint is the 215-tensor F32 release
//! `MelodyMachine/Deepfake-audio-detection-V2` at immutable revision
//! `de3cde5a29c449bb5268814e421b46bf6ebdcd72`. Its official configuration
//! declares `Wav2Vec2ForSequenceClassification`, not WavLM. The forward is:
//!
//! ```text
//! normalized 16 kHz mono PCM
//!   -> Wav2Vec2-base seven-convolution frontend
//!   -> 12 post-norm Transformer blocks
//!   -> per-frame Linear(768, 256)
//!   -> mean over time
//!   -> Linear(256, 2)
//!   -> logits in [fake, real] order
//! ```
//!
//! The Wav2Vec2 encoder is shared with the independently parity-tested CTC
//! implementation. Every learned convolution, matrix multiplication,
//! LayerNorm, GELU and softmax in that graph is routed through [`Compute`].
//! Only CPU and Metal are accepted here; any other backend fails explicitly
//! before inference and is never replaced with a silent CPU execution.

use std::path::Path;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, chunks};
use vokra_core::{CompliancePolicy, LicenseClass, Result, VokraError, check_weight_license};

use crate::align::charsiu::linear_forward_with_compute;
use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{
    StrictCheckpoint, StrictCheckpointSpec, load_tensor, require_tensor_shape,
};
use crate::wav2vec2_ctc::{WAV2VEC2_CTC_HOT_OPS, Wav2Vec2Ctc, reject_non_finite};

/// Converter/runtime architecture handshake.
pub const ARCH: &str = "deepfake_detection";
/// Canonical public model name.
pub const NAME: &str = "deepfake-audio-detection-v2";
/// Model-zoo task category.
pub const CATEGORY: &str = "classification";
/// Immutable upstream repository.
pub const UPSTREAM_HF: &str = "MelodyMachine/Deepfake-audio-detection-V2";
/// Immutable upstream checkpoint revision.
pub const UPSTREAM_REVISION: &str = "de3cde5a29c449bb5268814e421b46bf6ebdcd72";
/// Canonical checkpoint filename.
pub const CHECKPOINT_FILE: &str = "model.safetensors";
/// Canonical checkpoint SHA-256.
pub const CHECKPOINT_SHA256: &str =
    "997d9ce59e63151d5e444a6fa7c863986d0e56d515f67321bd705ac3b01bc38c";
/// Canonical config SHA-256.
pub const CONFIG_SHA256: &str = "a7ff31ca7ba4dc7fb5c4847d6dff0cb8daa1f0ec512e6ff8190664874c5b2806";
/// Canonical preprocessor-config SHA-256.
pub const PREPROCESSOR_SHA256: &str =
    "8cdfd65ff4115423185a1512bdae100e2e0cd744f5b322417429944aaafd0827";
/// Owner-signed weight licence.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

/// Required mono PCM sample rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// Wav2Vec2 residual width.
pub const HIDDEN_SIZE: usize = 768;
/// Wav2Vec2 Transformer depth.
pub const NUM_HIDDEN_LAYERS: usize = 12;
/// Wav2Vec2 attention-head count.
pub const NUM_ATTENTION_HEADS: usize = 12;
/// Wav2Vec2 feed-forward width.
pub const INTERMEDIATE_SIZE: usize = 3_072;
/// Sequence-classifier projector width.
pub const CLASSIFIER_PROJ_SIZE: usize = 256;
/// Binary output width.
pub const N_CLASSES: u32 = 2;
/// Official class order from the pinned config.
pub const CLASS_LABELS: [&str; 2] = ["fake", "real"];
/// Metadata key carrying [`CLASS_LABELS`].
pub const GGUF_KEY_ID2LABEL: &str = "vokra.deepfake.id2label";
/// Metadata key carrying the immutable upstream repository.
pub const GGUF_KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
/// Metadata key carrying the model category.
pub const GGUF_KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Official checkpoint page.
pub const PRIMARY_SOURCE_HF: &str =
    "https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2";
/// Exact Transformers source family used by the checkpoint.
pub const PRIMARY_SOURCE_TRANSFORMERS_CODE: &str = "https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py";
/// Historical public API alias retained after correcting the WavLM mistake.
#[deprecated(note = "the checkpoint is Wav2Vec2; use PRIMARY_SOURCE_TRANSFORMERS_CODE")]
pub const PRIMARY_SOURCE_WAVLM_CODE: &str = PRIMARY_SOURCE_TRANSFORMERS_CODE;
/// Historical public API alias retained after correcting the WavLM mistake.
#[deprecated(note = "the checkpoint is Wav2Vec2; use PRIMARY_SOURCE_TRANSFORMERS_CODE")]
pub const PRIMARY_SOURCE_WAVLM_PAPER: &str = "https://arxiv.org/abs/2006.11477";
/// Historical candidate list retained for source compatibility. Binding is
/// now exact and accepts only `classifier.weight`.
#[deprecated(note = "the canonical 215-tensor manifest pins classifier.weight exactly")]
pub const CLASSIFIER_WEIGHT_CANDIDATES: [&str; 4] = [
    "classifier.weight",
    "classifier.dense.weight",
    "model.classifier.weight",
    "wavlm.classifier.weight",
];
/// Historical loud-partial marker retained for source compatibility.
#[deprecated(note = "no primitive is missing; the complete Wav2Vec2 forward is implemented")]
pub const MISSING_PRIMITIVE: &str = "none (native Wav2Vec2 forward implemented)";

const TENSOR_COUNT: usize = 215;
const LAYER_NORM_EPS: f32 = 1.0e-5;
const MIN_PCM_SAMPLES: usize = 400;
const PREFIX: &str = "vokra.deepfake";
const CONV_DIM: [usize; 7] = [512; 7];
const CONV_KERNEL: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
const CONV_STRIDE: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];
const MANIFEST_SHA256: [u8; 32] = [
    0x81, 0xd7, 0x52, 0xd0, 0x0a, 0xbe, 0x58, 0x4c, 0x21, 0x60, 0xe9, 0xba, 0x93, 0x4c, 0x37, 0x8d,
    0x3e, 0xc7, 0x56, 0x01, 0x04, 0xbc, 0x78, 0xca, 0xe8, 0xb3, 0x38, 0x80, 0x9e, 0xc3, 0x1e, 0xbe,
];

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: NAME,
    arch: ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: TENSOR_COUNT,
    manifest_sha256: MANIFEST_SHA256,
};

/// Every learned hot operation used by the encoder and classifier head.
pub const DEEPFAKE_HOT_OPS: &[HotOp] = WAV2VEC2_CTC_HOT_OPS;

/// Fixed topology resolved from the immutable upstream configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepfakeDetectionConfig {
    pub sample_rate: u32,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub classifier_proj_size: usize,
    pub num_classes: usize,
    pub layer_norm_eps: f32,
}

impl Default for DeepfakeDetectionConfig {
    fn default() -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            hidden_size: HIDDEN_SIZE,
            num_hidden_layers: NUM_HIDDEN_LAYERS,
            num_attention_heads: NUM_ATTENTION_HEADS,
            intermediate_size: INTERMEDIATE_SIZE,
            classifier_proj_size: CLASSIFIER_PROJ_SIZE,
            num_classes: N_CLASSES as usize,
            layer_norm_eps: LAYER_NORM_EPS,
        }
    }
}

/// The detector's two raw logits in [`CLASS_LABELS`] order.
///
/// No default verdict threshold is embedded. Threshold selection is a
/// deployment decision and remains explicit through [`Self::exceeds`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepfakeScore {
    logits: [f32; 2],
}

impl DeepfakeScore {
    #[must_use]
    pub const fn from_logits(logits: [f32; 2]) -> Self {
        Self { logits }
    }

    #[must_use]
    pub const fn logits(&self) -> [f32; 2] {
        self.logits
    }

    /// Numerically stable binary softmax in `[fake, real]` order.
    #[must_use]
    pub fn probabilities(&self) -> [f32; 2] {
        let maximum = self.logits[0].max(self.logits[1]);
        let first = (self.logits[0] - maximum).exp();
        let second = (self.logits[1] - maximum).exp();
        let sum = first + second;
        [first / sum, second / sum]
    }

    pub fn probability_of(&self, index: usize) -> Result<f32> {
        if index >= N_CLASSES as usize {
            return Err(VokraError::InvalidArgument(format!(
                "deepfake_detection: class index {index} is out of range for {N_CLASSES} classes; valid indices are 0=fake and 1=real"
            )));
        }
        Ok(self.probabilities()[index])
    }

    pub fn exceeds(&self, index: usize, threshold: f32) -> Result<bool> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(VokraError::InvalidArgument(format!(
                "deepfake_detection: threshold must be finite and within [0, 1], got {threshold}"
            )));
        }
        Ok(self.probability_of(index)? > threshold)
    }
}

/// Strictly bound task-head weights plus the complete tensor manifest view.
#[derive(Debug, Clone)]
pub struct DeepfakeDetectionWeights {
    tensors: Vec<(String, Vec<usize>)>,
    classifier_weight: String,
    classifier_bias: Option<String>,
    hidden_size: usize,
    projector_weight: Vec<f32>,
    projector_bias: Vec<f32>,
    classifier_weight_values: Vec<f32>,
    classifier_bias_values: Vec<f32>,
}

impl DeepfakeDetectionWeights {
    /// Standalone strict head binder retained for the existing public API.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        StrictCheckpoint::bind(file, SPEC)?;
        validate_canonical_dtypes(file)?;
        Self::bind_head(file)
    }

    fn bind_head(file: &GgufFile) -> Result<Self> {
        require_tensor_shape(file, NAME, "wav2vec2.masked_spec_embed", &[HIDDEN_SIZE])?;
        let tensors = file
            .tensors()
            .iter()
            .map(|tensor| {
                (
                    tensor.name.clone(),
                    tensor
                        .dimensions
                        .iter()
                        .map(|&dimension| dimension as usize)
                        .collect(),
                )
            })
            .collect();
        Ok(Self {
            tensors,
            classifier_weight: "classifier.weight".to_owned(),
            classifier_bias: Some("classifier.bias".to_owned()),
            hidden_size: CLASSIFIER_PROJ_SIZE,
            projector_weight: load_tensor(
                file,
                NAME,
                "projector.weight",
                &[CLASSIFIER_PROJ_SIZE, HIDDEN_SIZE],
            )?,
            projector_bias: load_tensor(file, NAME, "projector.bias", &[CLASSIFIER_PROJ_SIZE])?,
            classifier_weight_values: load_tensor(
                file,
                NAME,
                "classifier.weight",
                &[N_CLASSES as usize, CLASSIFIER_PROJ_SIZE],
            )?,
            classifier_bias_values: load_tensor(
                file,
                NAME,
                "classifier.bias",
                &[N_CLASSES as usize],
            )?,
        })
    }

    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    #[must_use]
    pub fn classifier_weight_name(&self) -> &str {
        &self.classifier_weight
    }

    #[must_use]
    pub fn classifier_bias_name(&self) -> Option<&str> {
        self.classifier_bias.as_deref()
    }

    /// Input width of `classifier.weight` (the preceding projector width).
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    #[must_use]
    pub fn tensor_dims(&self, name: &str) -> Option<&[usize]> {
        self.tensors
            .iter()
            .find(|(tensor_name, _)| tensor_name == name)
            .map(|(_, dimensions)| dimensions.as_slice())
    }
}

/// Complete native Wav2Vec2 deepfake classifier.
#[derive(Debug, Clone)]
pub struct DeepfakeDetection {
    checkpoint: StrictCheckpoint,
    config: DeepfakeDetectionConfig,
    encoder: Wav2Vec2Ctc,
    weights: DeepfakeDetectionWeights,
    backend: BackendKind,
    category: String,
    upstream_hf: String,
    legacy_metadata_repaired: bool,
}

#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code))]
struct DeepfakeForward {
    encoder_features: Vec<f32>,
    projected_features: Vec<f32>,
    pooled_embedding: Vec<f32>,
    frames: usize,
    score: DeepfakeScore,
}

impl DeepfakeDetection {
    /// Opens and strictly binds a GGUF under the default compliance policy.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = GgufFile::open(path)?;
        Self::from_gguf(&file)
    }

    /// Strictly binds an already-open GGUF under the default policy.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Self::from_gguf_with_policy(file, &CompliancePolicy::strict())
    }

    /// Strictly binds under an explicit compliance policy.
    pub fn from_gguf_with_policy(file: &GgufFile, policy: &CompliancePolicy) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        validate_canonical_dtypes(file)?;
        let license = check_weight_license(file, policy)?;
        if license.class != LicenseClass::Permissive
            || checkpoint.weight_license() != LicenseClass::Permissive
        {
            return Err(VokraError::ModelLoad(format!(
                "deepfake_detection: weight license resolves to {}, expected Permissive for the owner-signed Apache-2.0 release",
                license.class.as_str()
            )));
        }
        let legacy_metadata_repaired = validate_metadata(file)?;
        let weights = DeepfakeDetectionWeights::bind_head(file)?;
        let encoder = Wav2Vec2Ctc::from_deepfake_file(file)?;
        Ok(Self {
            checkpoint,
            config: DeepfakeDetectionConfig::default(),
            encoder,
            weights,
            backend: BackendKind::Cpu,
            category: CATEGORY.to_owned(),
            upstream_hf: UPSTREAM_HF.to_owned(),
            legacy_metadata_repaired,
        })
    }

    /// Convenience strict binder with an explicit backend selection.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        Ok(Self::from_gguf(file)?.with_backend(backend))
    }

    /// Selects the execution backend. Unsupported values fail at inference.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.encoder = self.encoder.with_backend(backend);
        self.backend = backend;
        self
    }

    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    #[must_use]
    pub const fn config(&self) -> &DeepfakeDetectionConfig {
        &self.config
    }

    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    /// Existing optional return shape retained for source compatibility.
    #[must_use]
    pub fn model_name(&self) -> Option<&str> {
        Some(self.checkpoint.model_name())
    }

    /// Existing optional return shape retained for source compatibility.
    #[must_use]
    pub fn category(&self) -> Option<&str> {
        Some(&self.category)
    }

    /// Existing optional return shape retained for source compatibility.
    #[must_use]
    pub fn upstream_hf(&self) -> Option<&str> {
        Some(&self.upstream_hf)
    }

    #[must_use]
    pub const fn weights(&self) -> &DeepfakeDetectionWeights {
        &self.weights
    }

    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.checkpoint.tensor_count()
    }

    /// Input width of the final classifier (`256`).
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        CLASSIFIER_PROJ_SIZE
    }

    #[must_use]
    pub const fn num_classes() -> u32 {
        N_CLASSES
    }

    #[must_use]
    pub const fn class_labels() -> &'static [&'static str; 2] {
        &CLASS_LABELS
    }

    /// The canonical synthetic/fake class index from the pinned config.
    pub const fn spoof_class_index(&self) -> Result<usize> {
        Ok(0)
    }

    #[must_use]
    pub const fn legacy_metadata_repaired(&self) -> bool {
        self.legacy_metadata_repaired
    }

    /// Runs the normalized waveform frontend and Wav2Vec2 encoder.
    pub fn encode_features(&self, pcm: &[f32], sample_rate: u32) -> Result<(Vec<f32>, usize)> {
        validate_pcm(pcm, sample_rate)?;
        self.ensure_supported_backend()?;
        self.encoder.encode_features(pcm)
    }

    /// Scores mono PCM at an explicit sample rate.
    pub fn score_pcm(&self, pcm: &[f32], sample_rate: u32) -> Result<DeepfakeScore> {
        Ok(self.forward(pcm, sample_rate)?.score)
    }

    fn forward(&self, pcm: &[f32], sample_rate: u32) -> Result<DeepfakeForward> {
        let (features, frames) = self.encode_features(pcm, sample_rate)?;
        let compute = self.compute()?;
        let projected = linear_forward_with_compute(
            &features,
            frames,
            HIDDEN_SIZE,
            &self.weights.projector_weight,
            &self.weights.projector_bias,
            CLASSIFIER_PROJ_SIZE,
            &compute,
        )?;
        let pooled = mean_frames(&projected, frames, CLASSIFIER_PROJ_SIZE)?;
        let logits = linear_forward_with_compute(
            &pooled,
            1,
            CLASSIFIER_PROJ_SIZE,
            &self.weights.classifier_weight_values,
            &self.weights.classifier_bias_values,
            N_CLASSES as usize,
            &compute,
        )?;
        reject_non_finite("deepfake logits", &logits)?;
        Ok(DeepfakeForward {
            encoder_features: features,
            projected_features: projected,
            pooled_embedding: pooled,
            frames,
            score: DeepfakeScore::from_logits([logits[0], logits[1]]),
        })
    }

    /// Preserves the original implicit-16 kHz scoring API.
    pub fn score(&self, pcm: &[f32]) -> Result<DeepfakeScore> {
        self.score_pcm(pcm, SAMPLE_RATE)
    }

    fn ensure_supported_backend(&self) -> Result<()> {
        match self.backend {
            BackendKind::Cpu | BackendKind::Metal => Ok(()),
            other => Err(VokraError::UnsupportedOp(format!(
                "deepfake_detection: backend {other:?} is unsupported; this model implements only Mac CPU and Metal. Vokra will not silently execute learned operations on the CPU (FR-EX-08)"
            ))),
        }
    }

    fn compute(&self) -> Result<Compute> {
        self.ensure_supported_backend()?;
        Compute::for_backend(self.backend, DEEPFAKE_HOT_OPS)
    }
}

fn validate_pcm(pcm: &[f32], sample_rate: u32) -> Result<()> {
    if sample_rate != SAMPLE_RATE {
        return Err(VokraError::InvalidArgument(format!(
            "deepfake_detection: expected {SAMPLE_RATE} Hz mono PCM, got {sample_rate} Hz; resampling is caller-owned"
        )));
    }
    if pcm.len() < MIN_PCM_SAMPLES {
        return Err(VokraError::InvalidArgument(format!(
            "deepfake_detection: PCM has {} samples, but the seven-layer Wav2Vec2 frontend requires at least {MIN_PCM_SAMPLES}",
            pcm.len()
        )));
    }
    if let Some((index, value)) = pcm
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "deepfake_detection: PCM sample {index} is non-finite ({value})"
        )));
    }
    Ok(())
}

fn mean_frames(values: &[f32], frames: usize, width: usize) -> Result<Vec<f32>> {
    if frames == 0 || width == 0 || values.len() != frames * width {
        return Err(VokraError::InvalidArgument(format!(
            "deepfake_detection: mean-pool shape mismatch: values={}, frames={frames}, width={width}",
            values.len()
        )));
    }
    let mut pooled = vec![0.0f32; width];
    for frame in values.chunks_exact(width) {
        for (destination, value) in pooled.iter_mut().zip(frame) {
            *destination += *value;
        }
    }
    let inverse = 1.0 / frames as f32;
    for value in &mut pooled {
        *value *= inverse;
    }
    Ok(pooled)
}

fn validate_canonical_dtypes(file: &GgufFile) -> Result<()> {
    for tensor in file.tensors() {
        if tensor.dtype != GgmlType::F32 {
            return Err(VokraError::ModelLoad(format!(
                "deepfake_detection: tensor {:?} has {:?}, expected canonical F32",
                tensor.name, tensor.dtype
            )));
        }
    }
    Ok(())
}

const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_CHECKPOINT_FILE: &str = "vokra.provenance.checkpoint_file";
const KEY_CHECKPOINT_SHA256: &str = "vokra.provenance.checkpoint_sha256";
const KEY_CONFIG_SHA256: &str = "vokra.provenance.config_sha256";
const KEY_PREPROCESSOR_SHA256: &str = "vokra.provenance.preprocessor_sha256";
const LEGACY_SOURCE: &str = "MelodyMachine/Deepfake-audio-detection-V2 (WavLM binary classifier for audio deepfake detection, apache-2.0)";
const CANONICAL_SOURCE: &str = concat!(
    "MelodyMachine/Deepfake-audio-detection-V2@",
    "de3cde5a29c449bb5268814e421b46bf6ebdcd72",
    "/model.safetensors sha256:",
    "997d9ce59e63151d5e444a6fa7c863986d0e56d515f67321bd705ac3b01bc38c"
);

const PROVENANCE_KEYS: &[&str] = &[
    KEY_UPSTREAM_REVISION,
    KEY_CHECKPOINT_FILE,
    KEY_CHECKPOINT_SHA256,
    KEY_CONFIG_SHA256,
    KEY_PREPROCESSOR_SHA256,
];

const CONTRACT_KEYS: &[&str] = &[
    "vokra.deepfake.architecture",
    "vokra.deepfake.model_type",
    "vokra.deepfake.sample_rate",
    "vokra.deepfake.normalize",
    "vokra.deepfake.return_attention_mask",
    "vokra.deepfake.hidden_size",
    "vokra.deepfake.num_hidden_layers",
    "vokra.deepfake.num_attention_heads",
    "vokra.deepfake.intermediate_size",
    "vokra.deepfake.classifier_proj_size",
    "vokra.deepfake.num_classes",
    "vokra.deepfake.layer_norm_eps",
    "vokra.deepfake.feat_extract_norm",
    "vokra.deepfake.do_stable_layer_norm",
    "vokra.deepfake.hidden_act",
    "vokra.deepfake.num_conv_pos_embeddings",
    "vokra.deepfake.num_conv_pos_embedding_groups",
    "vokra.deepfake.use_weighted_layer_sum",
    "vokra.deepfake.conv_dim",
    "vokra.deepfake.conv_kernel",
    "vokra.deepfake.conv_stride",
    GGUF_KEY_ID2LABEL,
];

/// Validates exact generic provenance and the all-or-nothing additive groups.
/// Returns true only for the immutable historical public GGUF.
fn validate_metadata(file: &GgufFile) -> Result<bool> {
    require_string(file, GGUF_KEY_MODEL_CATEGORY, CATEGORY)?;
    require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, NAME)?;
    require_string(file, GGUF_KEY_UPSTREAM_HF, UPSTREAM_HF)?;
    let license = required_string(file, chunks::KEY_PROVENANCE_LICENSE)?;
    if !license.eq_ignore_ascii_case(DEFAULT_LICENSE_SPDX) {
        return Err(metadata_error(
            chunks::KEY_PROVENANCE_LICENSE,
            license,
            DEFAULT_LICENSE_SPDX,
        ));
    }
    let class = required_string(file, chunks::KEY_PROVENANCE_WEIGHT_LICENSE)?;
    if LicenseClass::from_class_str(class) != Some(LicenseClass::Permissive) {
        return Err(metadata_error(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            class,
            "permissive",
        ));
    }
    let source = required_string(file, chunks::KEY_PROVENANCE_SOURCE)?;
    if source != LEGACY_SOURCE && source != CANONICAL_SOURCE {
        return Err(VokraError::ModelLoad(format!(
            "deepfake_detection: unsupported `{}`={source:?}; expected historical {LEGACY_SOURCE:?} or canonical {CANONICAL_SOURCE:?}",
            chunks::KEY_PROVENANCE_SOURCE
        )));
    }

    let provenance_count = count_present(file, PROVENANCE_KEYS);
    let contract_count = count_present(file, CONTRACT_KEYS);
    match (provenance_count, contract_count) {
        (0, 0) => Ok(true),
        (provenance, contract)
            if provenance == PROVENANCE_KEYS.len() && contract == CONTRACT_KEYS.len() =>
        {
            require_string(file, KEY_UPSTREAM_REVISION, UPSTREAM_REVISION)?;
            require_string(file, KEY_CHECKPOINT_FILE, CHECKPOINT_FILE)?;
            require_string(file, KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256)?;
            require_string(file, KEY_CONFIG_SHA256, CONFIG_SHA256)?;
            require_string(file, KEY_PREPROCESSOR_SHA256, PREPROCESSOR_SHA256)?;
            validate_contract(file)?;
            Ok(false)
        }
        _ => Err(VokraError::ModelLoad(format!(
            "deepfake_detection: partial immutable metadata: provenance {provenance_count}/{}, `{PREFIX}.*` contract {contract_count}/{}; refusing topology repair",
            PROVENANCE_KEYS.len(),
            CONTRACT_KEYS.len()
        ))),
    }
}

fn validate_contract(file: &GgufFile) -> Result<()> {
    require_string(
        file,
        "vokra.deepfake.architecture",
        "Wav2Vec2ForSequenceClassification",
    )?;
    require_string(file, "vokra.deepfake.model_type", "wav2vec2")?;
    require_u64(file, "vokra.deepfake.sample_rate", SAMPLE_RATE as u64)?;
    require_bool(file, "vokra.deepfake.normalize", true)?;
    require_bool(file, "vokra.deepfake.return_attention_mask", false)?;
    require_u64(file, "vokra.deepfake.hidden_size", HIDDEN_SIZE as u64)?;
    require_u64(
        file,
        "vokra.deepfake.num_hidden_layers",
        NUM_HIDDEN_LAYERS as u64,
    )?;
    require_u64(
        file,
        "vokra.deepfake.num_attention_heads",
        NUM_ATTENTION_HEADS as u64,
    )?;
    require_u64(
        file,
        "vokra.deepfake.intermediate_size",
        INTERMEDIATE_SIZE as u64,
    )?;
    require_u64(
        file,
        "vokra.deepfake.classifier_proj_size",
        CLASSIFIER_PROJ_SIZE as u64,
    )?;
    require_u64(file, "vokra.deepfake.num_classes", N_CLASSES as u64)?;
    require_f64(
        file,
        "vokra.deepfake.layer_norm_eps",
        f64::from(LAYER_NORM_EPS),
    )?;
    require_string(file, "vokra.deepfake.feat_extract_norm", "group")?;
    require_bool(file, "vokra.deepfake.do_stable_layer_norm", false)?;
    require_string(file, "vokra.deepfake.hidden_act", "gelu")?;
    require_u64(file, "vokra.deepfake.num_conv_pos_embeddings", 128)?;
    require_u64(file, "vokra.deepfake.num_conv_pos_embedding_groups", 16)?;
    require_bool(file, "vokra.deepfake.use_weighted_layer_sum", false)?;
    require_u32_array(file, "vokra.deepfake.conv_dim", &CONV_DIM)?;
    require_u32_array(file, "vokra.deepfake.conv_kernel", &CONV_KERNEL)?;
    require_u32_array(file, "vokra.deepfake.conv_stride", &CONV_STRIDE)?;
    require_string_array(file, GGUF_KEY_ID2LABEL, &CLASS_LABELS)
}

fn count_present(file: &GgufFile, keys: &[&str]) -> usize {
    keys.iter().filter(|key| file.get(key).is_some()).count()
}

fn required_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!("deepfake_detection: missing/non-string `{key}`"))
        })
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = required_string(file, key)?;
    if actual != expected {
        return Err(metadata_error(key, actual, expected));
    }
    Ok(())
}

fn require_u64(file: &GgufFile, key: &str, expected: u64) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_u64)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!("deepfake_detection: missing/non-u32 `{key}`"))
        })?;
    if actual != expected {
        return Err(metadata_error(key, actual, expected));
    }
    Ok(())
}

fn require_f64(file: &GgufFile, key: &str, expected: f64) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_f64)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!("deepfake_detection: missing/non-f32 `{key}`"))
        })?;
    if actual.to_bits() != expected.to_bits() {
        return Err(metadata_error(key, actual, expected));
    }
    Ok(())
}

fn require_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_bool)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!("deepfake_detection: missing/non-bool `{key}`"))
        })?;
    if actual != expected {
        return Err(metadata_error(key, actual, expected));
    }
    Ok(())
}

fn require_u32_array(file: &GgufFile, key: &str, expected: &[usize]) -> Result<()> {
    let array = file
        .get(key)
        .and_then(GgufMetadataValue::as_array)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!("deepfake_detection: missing/non-array `{key}`"))
        })?;
    if array.values.len() != expected.len() {
        return Err(metadata_error(key, array.values.len(), expected.len()));
    }
    for (index, (actual, expected)) in array.values.iter().zip(expected).enumerate() {
        let actual = actual.as_u64().ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "deepfake_detection: `{key}` element {index} is not an unsigned integer"
            ))
        })?;
        if actual != *expected as u64 {
            return Err(VokraError::ModelLoad(format!(
                "deepfake_detection: `{key}` element {index}={actual}, expected {expected}"
            )));
        }
    }
    Ok(())
}

fn require_string_array(file: &GgufFile, key: &str, expected: &[&str]) -> Result<()> {
    let array = file
        .get(key)
        .and_then(GgufMetadataValue::as_array)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!("deepfake_detection: missing/non-array `{key}`"))
        })?;
    if array.values.len() != expected.len() {
        return Err(metadata_error(key, array.values.len(), expected.len()));
    }
    for (index, (actual, expected)) in array.values.iter().zip(expected).enumerate() {
        let actual = actual.as_str().ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "deepfake_detection: `{key}` element {index} is not a string"
            ))
        })?;
        if actual != *expected {
            return Err(VokraError::ModelLoad(format!(
                "deepfake_detection: `{key}` element {index}={actual:?}, expected {expected:?}"
            )));
        }
    }
    Ok(())
}

fn metadata_error(
    key: &str,
    actual: impl std::fmt::Debug,
    expected: impl std::fmt::Debug,
) -> VokraError {
    VokraError::ModelLoad(format!(
        "deepfake_detection: unsupported `{key}`={actual:?}; expected {expected:?}"
    ))
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use vokra_core::gguf::{GgufArray, GgufBuilder, GgufValueType};

    use super::*;

    fn metadata_builder() -> GgufBuilder {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        builder.add_string(chunks::KEY_MODEL_NAME, NAME);
        builder.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        builder.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, "permissive");
        builder.add_string(chunks::KEY_PROVENANCE_LICENSE, DEFAULT_LICENSE_SPDX);
        builder.add_string(chunks::KEY_PROVENANCE_MODEL_ID, NAME);
        builder.add_string(GGUF_KEY_UPSTREAM_HF, UPSTREAM_HF);
        builder.add_string(chunks::KEY_PROVENANCE_SOURCE, LEGACY_SOURCE);
        builder
    }

    fn add_u32_array(builder: &mut GgufBuilder, key: &str, values: &[usize]) {
        builder.add_metadata(
            key,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U32,
                values: values
                    .iter()
                    .map(|&value| GgufMetadataValue::U32(value as u32))
                    .collect(),
            }),
        );
    }

    fn add_string_array(builder: &mut GgufBuilder, key: &str, values: &[&str]) {
        builder.add_metadata(
            key,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::String,
                values: values
                    .iter()
                    .map(|value| GgufMetadataValue::String((*value).to_owned()))
                    .collect(),
            }),
        );
    }

    fn stamp_complete_metadata(builder: &mut GgufBuilder, labels: &[&str]) {
        builder.add_string(chunks::KEY_PROVENANCE_SOURCE, CANONICAL_SOURCE);
        builder.add_string(KEY_UPSTREAM_REVISION, UPSTREAM_REVISION);
        builder.add_string(KEY_CHECKPOINT_FILE, CHECKPOINT_FILE);
        builder.add_string(KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256);
        builder.add_string(KEY_CONFIG_SHA256, CONFIG_SHA256);
        builder.add_string(KEY_PREPROCESSOR_SHA256, PREPROCESSOR_SHA256);
        builder.add_string(
            "vokra.deepfake.architecture",
            "Wav2Vec2ForSequenceClassification",
        );
        builder.add_string("vokra.deepfake.model_type", "wav2vec2");
        builder.add_u32("vokra.deepfake.sample_rate", SAMPLE_RATE);
        builder.add_bool("vokra.deepfake.normalize", true);
        builder.add_bool("vokra.deepfake.return_attention_mask", false);
        builder.add_u32("vokra.deepfake.hidden_size", HIDDEN_SIZE as u32);
        builder.add_u32("vokra.deepfake.num_hidden_layers", NUM_HIDDEN_LAYERS as u32);
        builder.add_u32(
            "vokra.deepfake.num_attention_heads",
            NUM_ATTENTION_HEADS as u32,
        );
        builder.add_u32("vokra.deepfake.intermediate_size", INTERMEDIATE_SIZE as u32);
        builder.add_u32(
            "vokra.deepfake.classifier_proj_size",
            CLASSIFIER_PROJ_SIZE as u32,
        );
        builder.add_u32("vokra.deepfake.num_classes", N_CLASSES);
        builder.add_f32("vokra.deepfake.layer_norm_eps", LAYER_NORM_EPS);
        builder.add_string("vokra.deepfake.feat_extract_norm", "group");
        builder.add_bool("vokra.deepfake.do_stable_layer_norm", false);
        builder.add_string("vokra.deepfake.hidden_act", "gelu");
        builder.add_u32("vokra.deepfake.num_conv_pos_embeddings", 128);
        builder.add_u32("vokra.deepfake.num_conv_pos_embedding_groups", 16);
        builder.add_bool("vokra.deepfake.use_weighted_layer_sum", false);
        add_u32_array(builder, "vokra.deepfake.conv_dim", &CONV_DIM);
        add_u32_array(builder, "vokra.deepfake.conv_kernel", &CONV_KERNEL);
        add_u32_array(builder, "vokra.deepfake.conv_stride", &CONV_STRIDE);
        add_string_array(builder, GGUF_KEY_ID2LABEL, labels);
    }

    fn parse_metadata(builder: GgufBuilder) -> GgufFile {
        GgufFile::parse(builder.to_bytes().expect("serialize metadata GGUF"))
            .expect("parse metadata GGUF")
    }

    fn close(left: f32, right: f32) -> bool {
        (left - right).abs() < 1.0e-5
    }

    #[test]
    fn immutable_contract_is_wav2vec2_not_wavlm() {
        assert_eq!(TENSOR_COUNT, 215);
        assert_eq!(HIDDEN_SIZE, 768);
        assert_eq!(CLASSIFIER_PROJ_SIZE, 256);
        assert_eq!(CLASS_LABELS, ["fake", "real"]);
        assert_eq!(MANIFEST_SHA256.len(), 32);
        assert!(PRIMARY_SOURCE_TRANSFORMERS_CODE.contains("v4.41.2"));
    }

    #[test]
    fn exact_legacy_metadata_is_repaired_but_partial_groups_fail() {
        assert!(
            validate_metadata(&parse_metadata(metadata_builder()))
                .expect("exact historical metadata")
        );
        let mut partial = metadata_builder();
        partial.add_string(KEY_UPSTREAM_REVISION, UPSTREAM_REVISION);
        let error = validate_metadata(&parse_metadata(partial))
            .unwrap_err()
            .to_string();
        assert!(error.contains("partial immutable metadata"));
        assert!(error.contains("provenance 1/5"));
    }

    #[test]
    fn complete_metadata_and_label_order_are_enforced() {
        let mut canonical = metadata_builder();
        stamp_complete_metadata(&mut canonical, &CLASS_LABELS);
        assert!(!validate_metadata(&parse_metadata(canonical)).expect("canonical metadata"));

        let mut reordered = metadata_builder();
        stamp_complete_metadata(&mut reordered, &["real", "fake"]);
        let error = validate_metadata(&parse_metadata(reordered))
            .unwrap_err()
            .to_string();
        assert!(error.contains(GGUF_KEY_ID2LABEL));
        assert!(error.contains("element 0"));
    }

    #[test]
    fn score_softmax_is_stable_and_threshold_is_explicit() {
        let score = DeepfakeScore::from_logits([2.0, -1.0]);
        let probabilities = score.probabilities();
        assert!(probabilities[0] > probabilities[1]);
        assert!(close(probabilities[0] + probabilities[1], 1.0));
        assert!(score.exceeds(0, 0.5).unwrap());
        assert!(!score.exceeds(0, 0.99).unwrap());
        assert!(score.exceeds(0, f32::NAN).is_err());
        assert!(score.probability_of(2).is_err());

        let huge = DeepfakeScore::from_logits([200.0, 100.0]).probabilities();
        assert!(huge[0].is_finite() && huge[1].is_finite());
        assert!(close(huge[0] + huge[1], 1.0));
    }

    #[test]
    fn mean_pool_matches_transformers_sequence_classifier() {
        let pooled = mean_frames(&[1.0, 3.0, 5.0, 7.0], 2, 2).unwrap();
        assert_eq!(pooled, vec![3.0, 5.0]);
        assert!(mean_frames(&[1.0], 2, 2).is_err());
    }

    #[test]
    fn pcm_contract_is_explicit() {
        assert!(validate_pcm(&vec![0.0; MIN_PCM_SAMPLES], SAMPLE_RATE).is_ok());
        assert!(validate_pcm(&vec![0.0; MIN_PCM_SAMPLES], 48_000).is_err());
        assert!(validate_pcm(&vec![0.0; MIN_PCM_SAMPLES - 1], SAMPLE_RATE).is_err());
        let mut non_finite = vec![0.0; MIN_PCM_SAMPLES];
        non_finite[4] = f32::INFINITY;
        assert!(validate_pcm(&non_finite, SAMPLE_RATE).is_err());
    }

    #[test]
    fn learned_hot_ops_are_cpu_and_metal_complete() {
        Compute::for_backend(BackendKind::Cpu, DEEPFAKE_HOT_OPS)
            .expect("CPU covers every deepfake learned operation");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, DEEPFAKE_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("deepfake classifier has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    #[ignore = "requires VAST-prepared public GGUF and official Transformers fixture"]
    fn measure_official_cpu_against_transformers() {
        let (model, reference, pcm) = real_case(BackendKind::Cpu);
        let forward = model
            .forward(&pcm, SAMPLE_RATE)
            .expect("native deepfake CPU forward");
        measure_reference("cpu_vs_transformers", &reference, &pcm, &forward);
        eprintln!(
            "DEEPFAKE_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED"
        );
    }

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    #[ignore = "requires Apple Silicon, public GGUF and official Transformers fixture"]
    fn measure_official_metal_against_cpu_and_transformers() {
        if vokra_backend_metal::vokra_metal_probe().is_err() {
            eprintln!("skipping deepfake Metal measurement: no system Metal device");
            return;
        }
        let (cpu, reference, pcm) = real_case(BackendKind::Cpu);
        let cpu_forward = cpu
            .forward(&pcm, SAMPLE_RATE)
            .expect("native deepfake CPU forward");
        let (metal, _, _) = real_case(BackendKind::Metal);
        let metal_forward = metal
            .forward(&pcm, SAMPLE_RATE)
            .expect("native deepfake Metal forward");
        measure_reference("metal_vs_transformers", &reference, &pcm, &metal_forward);
        measure_forward_pair("metal_vs_cpu", &metal_forward, &cpu_forward);
        eprintln!(
            "DEEPFAKE_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED"
        );
    }

    fn real_case(backend: BackendKind) -> (DeepfakeDetection, PathBuf, Vec<f32>) {
        let gguf = std::env::var("VOKRA_DEEPFAKE_GGUF")
            .expect("VOKRA_DEEPFAKE_GGUF must point at the strict public GGUF");
        let reference = std::env::var("VOKRA_DEEPFAKE_REFERENCE_DIR")
            .expect("VOKRA_DEEPFAKE_REFERENCE_DIR must point at the official dump");
        let reference = Path::new(&reference).to_path_buf();
        let pcm = read_f32(&reference.join("input_pcm.f32"));
        assert_eq!(pcm.len(), SAMPLE_RATE as usize);
        let model = DeepfakeDetection::open(gguf)
            .expect("bind public deepfake GGUF")
            .with_backend(backend);
        (model, reference, pcm)
    }

    fn measure_reference(prefix: &str, reference: &Path, pcm: &[f32], forward: &DeepfakeForward) {
        let normalized = crate::wav2vec2_ctc::zero_mean_unit_var(pcm);
        let logits = forward.score.logits();
        let scores = forward.score.probabilities();
        assert_eq!(forward.encoder_features.len(), forward.frames * HIDDEN_SIZE);
        assert_eq!(
            forward.projected_features.len(),
            forward.frames * CLASSIFIER_PROJ_SIZE
        );
        for (name, actual) in [
            ("input_pcm", pcm),
            ("normalized_pcm", normalized.as_slice()),
            ("encoder_features", forward.encoder_features.as_slice()),
            ("projected_features", forward.projected_features.as_slice()),
            ("pooled_embedding", forward.pooled_embedding.as_slice()),
            ("logits", logits.as_slice()),
            ("scores", scores.as_slice()),
        ] {
            measure_pair(
                &format!("{prefix}/{name}"),
                actual,
                &read_f32(&reference.join(format!("{name}.f32"))),
            );
        }
    }

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    fn measure_forward_pair(prefix: &str, actual: &DeepfakeForward, expected: &DeepfakeForward) {
        let actual_logits = actual.score.logits();
        let expected_logits = expected.score.logits();
        let actual_scores = actual.score.probabilities();
        let expected_scores = expected.score.probabilities();
        for (name, left, right) in [
            (
                "encoder_features",
                actual.encoder_features.as_slice(),
                expected.encoder_features.as_slice(),
            ),
            (
                "projected_features",
                actual.projected_features.as_slice(),
                expected.projected_features.as_slice(),
            ),
            (
                "pooled_embedding",
                actual.pooled_embedding.as_slice(),
                expected.pooled_embedding.as_slice(),
            ),
            (
                "logits",
                actual_logits.as_slice(),
                expected_logits.as_slice(),
            ),
            (
                "scores",
                actual_scores.as_slice(),
                expected_scores.as_slice(),
            ),
        ] {
            measure_pair(&format!("{prefix}/{name}"), left, right);
        }
    }

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
        assert_eq!(bytes.len() % 4, 0, "unaligned f32 fixture {path:?}");
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    fn measure_pair(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        assert!(!actual.is_empty(), "{label} must not be empty");
        assert!(
            actual.iter().chain(expected).all(|value| value.is_finite()),
            "{label} must be finite"
        );
        let mut max_abs = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut dot = 0.0f64;
        let mut actual_sq = 0.0f64;
        let mut expected_sq = 0.0f64;
        let mut max_index = 0usize;
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let actual = f64::from(actual);
            let expected = f64::from(expected);
            let error = (actual - expected).abs();
            if error > max_abs {
                max_abs = error;
                max_index = index;
            }
            sum_abs += error;
            sum_sq += error * error;
            dot += actual * expected;
            actual_sq += actual * actual;
            expected_sq += expected * expected;
        }
        let count = actual.len() as f64;
        let norm_product = actual_sq.sqrt() * expected_sq.sqrt();
        let cosine = if norm_product == 0.0 {
            f64::NAN
        } else {
            dot / norm_product
        };
        eprintln!(
            "DEEPFAKE_MEASUREMENT label={label} elements={} max_abs={max_abs:.9e} \
             worst_index={max_index} mean_abs={:.9e} rms={:.9e} cosine={cosine:.12}",
            actual.len(),
            sum_abs / count,
            (sum_sq / count).sqrt(),
        );
    }
}
