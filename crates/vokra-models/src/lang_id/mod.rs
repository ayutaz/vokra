//! Native SpeechBrain ECAPA spoken-language identification.
//!
//! The two supported releases share an ECAPA family backbone but do not share
//! one complete topology:
//!
//! - `speechbrain/lang-id-voxlingua107-ecapa`: 60-bin frontend, 256-d
//!   embedding, XVector MLP classifier and log-softmax.
//! - `speechbrain/lang-id-commonlanguage_ecapa`: 80-bin frontend, 192-d
//!   embedding and cosine classifier.
//!
//! Conversion must use
//! `tools/parity/speechbrain_lang_id_prepare_checkpoint.py`. Its v2 contract
//! carries the complete official classifier, ordered labels and every
//! variant-specific ECAPA axis. Historical embedding-only GGUFs fail during
//! binding; no classifier, label inventory or frontend value is guessed.
//!
//! Learned ECAPA convolutions, attentive-pooling softmax and classifier GEMVs
//! all use [`Compute`]. Selecting Metal is therefore observable and an
//! unavailable operation is an explicit error, never a CPU fallback.

use std::collections::HashSet;

use vokra_core::gguf::{GgufFile, GgufMetadataValue, GgufValueType, chunks};
use vokra_core::{BackendKind, LicenseClass, Result, VokraError};
use vokra_ops::{SpeechbrainFbankAttrs, speechbrain_fbank};

use crate::compute::{Compute, HotOp};
use crate::ecapa_tdnn::{EcapaBackbone, EcapaBackboneConfig};

/// Shared converter/runtime arch tag.
pub const ARCH: &str = "lang_id_ecapa";
/// Official VoxLingua107 model identity.
pub const NAME_VOXLINGUA107: &str = "lang-id-voxlingua107-ecapa";
/// Official CommonLanguage model identity.
pub const NAME_COMMONLANGUAGE: &str = "lang-id-commonlanguage-ecapa";
/// Model task category.
pub const CATEGORY: &str = "classification";
/// Official VoxLingua107 upstream repository.
pub const UPSTREAM_HF_VOXLINGUA107: &str = "speechbrain/lang-id-voxlingua107-ecapa";
/// Official CommonLanguage upstream repository.
pub const UPSTREAM_HF_COMMONLANGUAGE: &str = "speechbrain/lang-id-commonlanguage_ecapa";

/// Prefix of the complete 200-tensor ECAPA embedding module.
pub const TRUNK_PREFIX: &str = "embedding_model.";
/// Representative exact ECAPA tensor name used by diagnostics.
pub const TRUNK_EXAMPLE_TENSOR: &str = "embedding_model.blocks.0.conv.conv.weight";

/// GGUF metadata key carrying the model-zoo task category.
pub const GGUF_KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// GGUF metadata key carrying the upstream Hugging Face repository.
pub const GGUF_KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

const KEY_UPSTREAM_REVISION: &str = "vokra.lang_id.upstream_revision";
const KEY_SAMPLE_RATE: &str = "vokra.lang_id.sample_rate";
const KEY_N_MELS: &str = "vokra.lang_id.n_mels";
const KEY_TDNN_CHANNELS: &str = "vokra.lang_id.tdnn_channels";
const KEY_MFA_CHANNELS: &str = "vokra.lang_id.mfa_channels";
const KEY_ATTENTION_CHANNELS: &str = "vokra.lang_id.attention_channels";
const KEY_RES2NET_SCALE: &str = "vokra.lang_id.res2net_scale";
const KEY_BLOCK_KERNELS: &str = "vokra.lang_id.block_kernels";
const KEY_BLOCK_DILATIONS: &str = "vokra.lang_id.block_dilations";
const KEY_EMBEDDING_DIM: &str = "vokra.lang_id.embedding_dim";
const KEY_CLASSIFIER_KIND: &str = "vokra.lang_id.classifier_kind";
const KEY_CLASSIFIER_HIDDEN_DIM: &str = "vokra.lang_id.classifier_hidden_dim";
const KEY_CLASS_COUNT: &str = "vokra.lang_id.class_count";
const KEY_LABELS: &str = "vokra.lang_id.labels";
const KEY_BN_EPS: &str = "vokra.lang_id.bn_eps";
const KEY_STATS_EPS: &str = "vokra.lang_id.stats_eps";
const KEY_LEAKY_RELU_SLOPE: &str = "vokra.lang_id.leaky_relu_slope";
const KEY_ARTIFACT_LAYOUT: &str = "vokra.lang_id.artifact_layout";
const ARTIFACT_LAYOUT: &str = "speechbrain-lang-id-prepared-v2";

const SHARED_TDNN_CHANNELS: usize = 1_024;
const SHARED_MFA_CHANNELS: usize = 3_072;
const SHARED_ATTENTION_CHANNELS: usize = 128;
const SHARED_RES2NET_SCALE: usize = 8;
const SHARED_DILATIONS: [usize; 3] = [2, 3, 4];
const BN_EPS: f32 = 1.0e-5;
const STATS_EPS: f32 = 1.0e-12;
const L2_EPS: f32 = 1.0e-12;
const BACKBONE_TENSORS: usize = 200;
const X_VECTOR_HEAD_TENSORS: usize = 12;
const COSINE_HEAD_TENSORS: usize = 1;
const LANG_ID_HOT_OPS: &[HotOp] = &[HotOp::Conv1d, HotOp::Softmax, HotOp::Gemv];

/// Which official SpeechBrain Lang-ID release the GGUF carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LangIdVariant {
    /// SpeechBrain VoxLingua107 ECAPA language classifier.
    VoxLingua107,
    /// SpeechBrain CommonLanguage ECAPA language classifier.
    CommonLanguage,
}

impl LangIdVariant {
    /// Resolves a canonical Vokra model name to its release variant.
    #[must_use]
    pub fn from_model_name(name: &str) -> Option<Self> {
        match name {
            NAME_VOXLINGUA107 => Some(Self::VoxLingua107),
            NAME_COMMONLANGUAGE => Some(Self::CommonLanguage),
            _ => None,
        }
    }

    /// Returns the canonical Vokra model name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::VoxLingua107 => NAME_VOXLINGUA107,
            Self::CommonLanguage => NAME_COMMONLANGUAGE,
        }
    }

    /// Returns the official upstream Hugging Face repository.
    #[must_use]
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::VoxLingua107 => UPSTREAM_HF_VOXLINGUA107,
            Self::CommonLanguage => UPSTREAM_HF_COMMONLANGUAGE,
        }
    }

    const fn n_mels(self) -> usize {
        match self {
            Self::VoxLingua107 => 60,
            Self::CommonLanguage => 80,
        }
    }

    const fn embedding_dim(self) -> usize {
        match self {
            Self::VoxLingua107 => 256,
            Self::CommonLanguage => 192,
        }
    }

    const fn block_kernels(self) -> [usize; 3] {
        match self {
            Self::VoxLingua107 => [3, 3, 3],
            Self::CommonLanguage => [3, 3, 1],
        }
    }

    const fn classifier_kind(self) -> &'static str {
        match self {
            Self::VoxLingua107 => "xvector-mlp-log-softmax-v1",
            Self::CommonLanguage => "ecapa-cosine-v1",
        }
    }

    const fn official_class_count(self) -> usize {
        match self {
            Self::VoxLingua107 => 107,
            Self::CommonLanguage => 45,
        }
    }
}

#[derive(Debug, Clone)]
struct LangIdContract {
    variant: LangIdVariant,
    upstream_revision: String,
    sample_rate: u32,
    labels: Vec<String>,
    classifier_hidden_dim: Option<usize>,
    leaky_relu_slope: Option<f32>,
    bn_eps: f32,
    stats_eps: f32,
}

impl LangIdContract {
    fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        let model_name = metadata_string(file, chunks::KEY_MODEL_NAME)?;
        let variant = LangIdVariant::from_model_name(model_name).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "lang_id_ecapa: unsupported `vokra.model.name` `{model_name}`; expected `{NAME_VOXLINGUA107}` or `{NAME_COMMONLANGUAGE}`"
            ))
        })?;
        require_string(file, GGUF_KEY_MODEL_CATEGORY, CATEGORY)?;
        require_string(file, GGUF_KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf())?;
        require_string(file, KEY_ARTIFACT_LAYOUT, ARTIFACT_LAYOUT)?;

        let upstream_revision = metadata_string(file, KEY_UPSTREAM_REVISION)?.to_owned();
        if upstream_revision.len() != 40
            || !upstream_revision
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(VokraError::ModelLoad(format!(
                "lang_id_ecapa: `{KEY_UPSTREAM_REVISION}` must be a full 40-hex commit"
            )));
        }

        require_u32(file, KEY_SAMPLE_RATE, 16_000)?;
        require_u32(file, KEY_N_MELS, variant.n_mels() as u32)?;
        require_u32(file, KEY_TDNN_CHANNELS, SHARED_TDNN_CHANNELS as u32)?;
        require_u32(file, KEY_MFA_CHANNELS, SHARED_MFA_CHANNELS as u32)?;
        require_u32(
            file,
            KEY_ATTENTION_CHANNELS,
            SHARED_ATTENTION_CHANNELS as u32,
        )?;
        require_u32(file, KEY_RES2NET_SCALE, SHARED_RES2NET_SCALE as u32)?;
        require_u32(file, KEY_EMBEDDING_DIM, variant.embedding_dim() as u32)?;
        require_u32_array(file, KEY_BLOCK_KERNELS, &variant.block_kernels())?;
        require_u32_array(file, KEY_BLOCK_DILATIONS, &SHARED_DILATIONS)?;
        require_string(file, KEY_CLASSIFIER_KIND, variant.classifier_kind())?;

        let class_count = require_u32_value(file, KEY_CLASS_COUNT)? as usize;
        if class_count != variant.official_class_count() {
            return Err(VokraError::ModelLoad(format!(
                "lang_id_ecapa: `{KEY_CLASS_COUNT}` is {class_count}, expected {} for `{}`",
                variant.official_class_count(),
                variant.name()
            )));
        }
        let labels = require_string_array(file, KEY_LABELS)?;
        if labels.len() != class_count {
            return Err(VokraError::ModelLoad(format!(
                "lang_id_ecapa: `{KEY_LABELS}` has {} entries, expected class_count={class_count}",
                labels.len()
            )));
        }
        let unique = labels.iter().collect::<HashSet<_>>();
        if unique.len() != labels.len() {
            return Err(VokraError::ModelLoad(
                "lang_id_ecapa: ordered label inventory contains duplicates".into(),
            ));
        }

        let bn_eps = require_f32(file, KEY_BN_EPS)?;
        let stats_eps = require_f32(file, KEY_STATS_EPS)?;
        require_exact_f32(KEY_BN_EPS, bn_eps, BN_EPS)?;
        require_exact_f32(KEY_STATS_EPS, stats_eps, STATS_EPS)?;

        let (classifier_hidden_dim, leaky_relu_slope) = match variant {
            LangIdVariant::VoxLingua107 => {
                let hidden = require_u32_value(file, KEY_CLASSIFIER_HIDDEN_DIM)? as usize;
                if hidden != 512 {
                    return Err(VokraError::ModelLoad(format!(
                        "lang_id_ecapa: `{KEY_CLASSIFIER_HIDDEN_DIM}` is {hidden}, expected official width 512"
                    )));
                }
                let slope = require_f32(file, KEY_LEAKY_RELU_SLOPE)?;
                require_exact_f32(KEY_LEAKY_RELU_SLOPE, slope, 0.01)?;
                (Some(hidden), Some(slope))
            }
            LangIdVariant::CommonLanguage => {
                reject_metadata(file, KEY_CLASSIFIER_HIDDEN_DIM)?;
                reject_metadata(file, KEY_LEAKY_RELU_SLOPE)?;
                (None, None)
            }
        };

        Ok(Self {
            variant,
            upstream_revision,
            sample_rate: 16_000,
            labels,
            classifier_hidden_dim,
            leaky_relu_slope,
            bn_eps,
            stats_eps,
        })
    }

    fn backbone_config(&self) -> EcapaBackboneConfig {
        EcapaBackboneConfig {
            input_dim: self.variant.n_mels(),
            tdnn_channels: SHARED_TDNN_CHANNELS,
            res2net_scale: SHARED_RES2NET_SCALE,
            mfa_channels: SHARED_MFA_CHANNELS,
            attention_channels: SHARED_ATTENTION_CHANNELS,
            embedding_dim: self.variant.embedding_dim(),
            block_kernels: self.variant.block_kernels(),
            block_dilations: SHARED_DILATIONS,
            bn_eps: self.bn_eps,
            stats_eps: self.stats_eps,
            tensor_prefix: TRUNK_PREFIX,
            diagnostic: ARCH,
        }
    }
}

/// Disk manifest retained for diagnostics and compatibility with the earlier
/// binder accessors. Successful model binding now requires the exact complete
/// manifest rather than using this list to infer a topology.
#[derive(Debug, Clone)]
pub struct LangIdWeights {
    tensors: Vec<(String, Vec<usize>)>,
}

impl LangIdWeights {
    /// Captures and validates the complete GGUF tensor manifest.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
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
            .collect::<Vec<_>>();
        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(
                "lang_id_ecapa: GGUF carries zero tensors; historical metadata-only and embedding-only artifacts are unsupported (FR-EX-08)"
                    .into(),
            ));
        }
        if !tensors.iter().any(|(name, _)| name == TRUNK_EXAMPLE_TENSOR) {
            return Err(VokraError::ModelLoad(format!(
                "lang_id_ecapa: missing required ECAPA tensor `{TRUNK_EXAMPLE_TENSOR}`"
            )));
        }
        Ok(Self { tensors })
    }

    /// Returns the number of tensors in the retained disk manifest.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Returns tensor names in GGUF manifest order.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// Returns dimensions for one exact tensor name.
    #[must_use]
    pub fn tensor_dims(&self, name: &str) -> Option<&[usize]> {
        self.tensors
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, dimensions)| dimensions.as_slice())
    }

    /// Counts tensors whose names begin with `prefix`.
    #[must_use]
    pub fn count_with_prefix(&self, prefix: &str) -> usize {
        self.tensors
            .iter()
            .filter(|(name, _)| name.starts_with(prefix))
            .count()
    }

    /// Returns all authenticated ECAPA trunk tensors and dimensions.
    #[must_use]
    pub fn trunk_tensors(&self) -> Vec<(&str, &[usize])> {
        self.tensors
            .iter()
            .filter(|(name, _)| name.starts_with(TRUNK_PREFIX))
            .map(|(name, dimensions)| (name.as_str(), dimensions.as_slice()))
            .collect()
    }

    /// Returns all authenticated language-classifier tensors and dimensions.
    #[must_use]
    pub fn language_head_tensors(&self) -> Vec<(&str, &[usize])> {
        self.tensors
            .iter()
            .filter(|(name, _)| name.starts_with("classifier."))
            .map(|(name, dimensions)| (name.as_str(), dimensions.as_slice()))
            .collect()
    }

    /// Derives the classifier language count from the disk head shape.
    #[must_use]
    pub fn language_count_from_disk(&self) -> Option<usize> {
        self.tensor_dims("classifier.output.weight")
            .or_else(|| self.tensor_dims("classifier.cosine.weight"))
            .and_then(|dimensions| dimensions.first().copied())
    }
}

#[derive(Debug)]
struct VectorBatchNorm {
    scale: Vec<f32>,
    shift: Vec<f32>,
}

impl VectorBatchNorm {
    fn bind(file: &GgufFile, prefix: &str, width: usize, eps: f32) -> Result<Self> {
        let gamma = tensor(file, &format!("{prefix}.weight"), &[width])?;
        let beta = tensor(file, &format!("{prefix}.bias"), &[width])?;
        let mean = tensor(file, &format!("{prefix}.running_mean"), &[width])?;
        let variance = tensor(file, &format!("{prefix}.running_var"), &[width])?;
        let mut scale = vec![0.0; width];
        let mut shift = vec![0.0; width];
        for index in 0..width {
            scale[index] = gamma[index] / (variance[index] + eps).sqrt();
            shift[index] = beta[index] - mean[index] * scale[index];
        }
        Ok(Self { scale, shift })
    }

    fn apply(&self, values: &mut [f32]) -> Result<()> {
        if values.len() != self.scale.len() {
            return Err(VokraError::InvalidArgument(format!(
                "lang_id_ecapa: BatchNorm input width {}, expected {}",
                values.len(),
                self.scale.len()
            )));
        }
        for ((value, &scale), &shift) in values.iter_mut().zip(&self.scale).zip(&self.shift) {
            *value = *value * scale + shift;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct XVectorHead {
    input_norm: VectorBatchNorm,
    hidden_weight: Vec<f32>,
    hidden_bias: Vec<f32>,
    hidden_norm: VectorBatchNorm,
    output_weight: Vec<f32>,
    output_bias: Vec<f32>,
    input_dim: usize,
    hidden_dim: usize,
    class_count: usize,
    leaky_relu_slope: f32,
}

impl XVectorHead {
    fn bind(file: &GgufFile, contract: &LangIdContract) -> Result<Self> {
        let input_dim = contract.variant.embedding_dim();
        let hidden_dim = contract.classifier_hidden_dim.ok_or_else(|| {
            VokraError::ModelLoad("lang_id_ecapa: XVector hidden width is absent".into())
        })?;
        let class_count = contract.labels.len();
        Ok(Self {
            input_norm: VectorBatchNorm::bind(
                file,
                "classifier.input_norm",
                input_dim,
                contract.bn_eps,
            )?,
            hidden_weight: tensor(file, "classifier.hidden.weight", &[hidden_dim, input_dim])?,
            hidden_bias: tensor(file, "classifier.hidden.bias", &[hidden_dim])?,
            hidden_norm: VectorBatchNorm::bind(
                file,
                "classifier.hidden_norm",
                hidden_dim,
                contract.bn_eps,
            )?,
            output_weight: tensor(file, "classifier.output.weight", &[class_count, hidden_dim])?,
            output_bias: tensor(file, "classifier.output.bias", &[class_count])?,
            input_dim,
            hidden_dim,
            class_count,
            leaky_relu_slope: contract.leaky_relu_slope.ok_or_else(|| {
                VokraError::ModelLoad("lang_id_ecapa: XVector LeakyReLU slope is absent".into())
            })?,
        })
    }

    fn forward(&self, embedding: &[f32], compute: &Compute) -> Result<Vec<f32>> {
        if embedding.len() != self.input_dim {
            return Err(VokraError::InvalidArgument(format!(
                "lang_id_ecapa: classifier embedding width {}, expected {}",
                embedding.len(),
                self.input_dim
            )));
        }
        let mut normalized = embedding.to_vec();
        leaky_relu(&mut normalized, self.leaky_relu_slope);
        self.input_norm.apply(&mut normalized)?;

        let mut hidden = vec![0.0; self.hidden_dim];
        compute.gemv_f32(
            self.hidden_dim,
            self.input_dim,
            &self.hidden_weight,
            &normalized,
            Some(&self.hidden_bias),
            &mut hidden,
        )?;
        leaky_relu(&mut hidden, self.leaky_relu_slope);
        self.hidden_norm.apply(&mut hidden)?;

        let mut output = vec![0.0; self.class_count];
        compute.gemv_f32(
            self.class_count,
            self.hidden_dim,
            &self.output_weight,
            &hidden,
            Some(&self.output_bias),
            &mut output,
        )?;
        log_softmax_in_place(&mut output)?;
        Ok(output)
    }
}

#[derive(Debug)]
struct CosineHead {
    normalized_weight: Vec<f32>,
    input_dim: usize,
    class_count: usize,
}

impl CosineHead {
    fn bind(file: &GgufFile, contract: &LangIdContract) -> Result<Self> {
        let input_dim = contract.variant.embedding_dim();
        let class_count = contract.labels.len();
        let mut normalized_weight =
            tensor(file, "classifier.cosine.weight", &[class_count, input_dim])?;
        for row in normalized_weight.chunks_exact_mut(input_dim) {
            l2_normalize(row, L2_EPS)?;
        }
        Ok(Self {
            normalized_weight,
            input_dim,
            class_count,
        })
    }

    fn forward(&self, embedding: &[f32], compute: &Compute) -> Result<Vec<f32>> {
        if embedding.len() != self.input_dim {
            return Err(VokraError::InvalidArgument(format!(
                "lang_id_ecapa: cosine embedding width {}, expected {}",
                embedding.len(),
                self.input_dim
            )));
        }
        let mut normalized = embedding.to_vec();
        l2_normalize(&mut normalized, L2_EPS)?;
        let mut output = vec![0.0; self.class_count];
        compute.gemv_f32(
            self.class_count,
            self.input_dim,
            &self.normalized_weight,
            &normalized,
            None,
            &mut output,
        )?;
        Ok(output)
    }
}

#[derive(Debug)]
enum ClassifierHead {
    XVector(XVectorHead),
    Cosine(CosineHead),
}

impl ClassifierHead {
    fn bind(file: &GgufFile, contract: &LangIdContract) -> Result<Self> {
        let classifier_count = file
            .tensors()
            .iter()
            .filter(|tensor| tensor.name.starts_with("classifier."))
            .count();
        let expected = match contract.variant {
            LangIdVariant::VoxLingua107 => X_VECTOR_HEAD_TENSORS,
            LangIdVariant::CommonLanguage => COSINE_HEAD_TENSORS,
        };
        if classifier_count != expected || file.tensors().len() != BACKBONE_TENSORS + expected {
            return Err(VokraError::ModelLoad(format!(
                "lang_id_ecapa: complete prepared-v2 manifest requires {BACKBONE_TENSORS} embedding + {expected} classifier tensors; found total={} classifier={classifier_count}. Historical embedding-only GGUFs are unsupported (FR-EX-08)",
                file.tensors().len()
            )));
        }
        match contract.variant {
            LangIdVariant::VoxLingua107 => Ok(Self::XVector(XVectorHead::bind(file, contract)?)),
            LangIdVariant::CommonLanguage => Ok(Self::Cosine(CosineHead::bind(file, contract)?)),
        }
    }

    fn forward(&self, embedding: &[f32], compute: &Compute) -> Result<Vec<f32>> {
        match self {
            Self::XVector(head) => head.forward(embedding, compute),
            Self::Cosine(head) => head.forward(embedding, compute),
        }
    }
}

/// Complete native Lang-ID runtime.
#[derive(Debug)]
pub struct LangIdEcapa {
    weights: LangIdWeights,
    backbone: EcapaBackbone,
    classifier: ClassifierHead,
    frontend: SpeechbrainFbankAttrs,
    contract: LangIdContract,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl LangIdEcapa {
    /// Strictly binds a complete prepared-v2 GGUF.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let contract = LangIdContract::from_gguf(file)?;
        let weights = LangIdWeights::from_gguf(file)?;
        let backbone = EcapaBackbone::bind(file, contract.backbone_config())?;
        let classifier = ClassifierHead::bind(file, &contract)?;
        let mut frontend = SpeechbrainFbankAttrs::ecapa_voxceleb();
        frontend.n_mels = contract.variant.n_mels();
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(GgufMetadataValue::as_str)
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        Ok(Self {
            weights,
            backbone,
            classifier,
            frontend,
            contract,
            weight_license,
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and strictly binds a GGUF file.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Selects the backend for every learned convolution, pooling softmax and
    /// classifier matrix-vector product.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the explicitly selected execution backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the authenticated release variant.
    #[must_use]
    pub const fn variant(&self) -> Option<LangIdVariant> {
        Some(self.contract.variant)
    }

    /// Returns the canonical model name.
    #[must_use]
    pub fn model_name(&self) -> Option<&str> {
        Some(self.contract.variant.name())
    }

    /// Returns the official upstream Hugging Face repository.
    #[must_use]
    pub fn upstream_hf(&self) -> Option<&str> {
        Some(self.contract.variant.upstream_hf())
    }

    /// Returns the pinned upstream checkpoint revision.
    #[must_use]
    pub fn upstream_revision(&self) -> &str {
        &self.contract.upstream_revision
    }

    /// Returns the stamped weight-license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Returns classifier labels in output-index order.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.contract.labels
    }

    /// Returns the authenticated number of language classes.
    #[must_use]
    pub fn language_count(&self) -> Option<usize> {
        Some(self.contract.labels.len())
    }

    /// Reports whether the complete language-classifier head is bound.
    #[must_use]
    pub const fn has_language_head(&self) -> bool {
        true
    }

    /// Returns the number of authenticated tensor descriptors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Returns the retained diagnostic tensor manifest.
    #[must_use]
    pub const fn weights(&self) -> &LangIdWeights {
        &self.weights
    }

    /// Runs the official 16 kHz frontend, complete ECAPA backbone and official
    /// classifier. VoxLingua107 returns log-posteriors; CommonLanguage returns
    /// the official cosine scores.
    pub fn identify(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        self.identify_pcm(pcm, self.contract.sample_rate)
    }

    /// Rate-explicit form of [`identify`](Self::identify).
    pub fn identify_pcm(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        if sample_rate != self.contract.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "lang_id_ecapa: expected {} Hz mono PCM, got {sample_rate} Hz; resample offline first",
                self.contract.sample_rate
            )));
        }
        let (features, frames) = speechbrain_fbank(pcm, &self.frontend)?;
        self.identify_features(&features, frames)
    }

    /// Computes only the pinned SpeechBrain frontend for independent parity.
    pub fn frontend_features(&self, pcm: &[f32], sample_rate: u32) -> Result<(Vec<f32>, usize)> {
        if sample_rate != self.contract.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "lang_id_ecapa: expected {} Hz mono PCM, got {sample_rate} Hz",
                self.contract.sample_rate
            )));
        }
        speechbrain_fbank(pcm, &self.frontend)
    }

    /// Runs only the complete ECAPA embedding module on row-major features.
    pub fn embed_features(&self, features: &[f32], frames: usize) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, LANG_ID_HOT_OPS)?;
        self.backbone.embed_features(features, frames, &compute)
    }

    /// Runs only the official classifier on one ECAPA embedding. This stage
    /// boundary exists so the independent SpeechBrain fixture can distinguish
    /// classifier drift from frontend or backbone drift.
    pub fn classify_embedding(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, LANG_ID_HOT_OPS)?;
        self.classifier.forward(embedding, &compute)
    }

    /// Runs a row-major `[frames, n_mels]` feature buffer. Exposed for the
    /// independent upstream parity fixture.
    pub fn identify_features(&self, features: &[f32], frames: usize) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, LANG_ID_HOT_OPS)?;
        let embedding = self.backbone.embed_features(features, frames, &compute)?;
        self.classifier.forward(&embedding, &compute)
    }

    /// Returns the highest-scoring label index and label.
    pub fn best_label<'a>(&'a self, scores: &[f32]) -> Result<(usize, &'a str)> {
        if scores.len() != self.contract.labels.len() || scores.is_empty() {
            return Err(VokraError::InvalidArgument(format!(
                "lang_id_ecapa: score width {}, expected {}",
                scores.len(),
                self.contract.labels.len()
            )));
        }
        let (index, _) = scores
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .ok_or_else(|| VokraError::InvalidArgument("lang_id_ecapa: empty scores".into()))?;
        Ok((index, self.contract.labels[index].as_str()))
    }
}

fn leaky_relu(values: &mut [f32], slope: f32) {
    for value in values {
        if *value < 0.0 {
            *value *= slope;
        }
    }
}

fn log_softmax_in_place(values: &mut [f32]) -> Result<()> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "lang_id_ecapa: cannot apply log-softmax to empty or non-finite logits".into(),
        ));
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = values
        .iter()
        .map(|value| (*value - maximum).exp())
        .sum::<f32>()
        .ln()
        + maximum;
    for value in values {
        *value -= log_sum;
    }
    Ok(())
}

fn l2_normalize(values: &mut [f32], eps: f32) -> Result<()> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(
            "lang_id_ecapa: cannot normalize empty or non-finite classifier values".into(),
        ));
    }
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    let denominator = norm.max(eps);
    for value in values {
        *value /= denominator;
    }
    Ok(())
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("lang_id_ecapa: missing tensor `{name}`")))?;
    let expected_u64 = expected
        .iter()
        .map(|&dimension| dimension as u64)
        .collect::<Vec<_>>();
    if info.dimensions != expected_u64 {
        return Err(VokraError::ModelLoad(format!(
            "lang_id_ecapa: tensor `{name}` has dims {:?}, expected {expected_u64:?}",
            info.dimensions
        )));
    }
    let values = file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("lang_id_ecapa: reading `{name}`: {error}"))
    })?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(format!(
            "lang_id_ecapa: tensor `{name}` contains non-finite values"
        )));
    }
    Ok(values)
}

fn metadata_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("lang_id_ecapa: missing string `{key}`")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = metadata_string(file, key)?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "lang_id_ecapa: `{key}` is `{actual}`, expected `{expected}`"
        )));
    }
    Ok(())
}

fn require_u32_value(file: &GgufFile, key: &str) -> Result<u32> {
    file.get(key)
        .and_then(GgufMetadataValue::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| VokraError::ModelLoad(format!("lang_id_ecapa: missing u32 `{key}`")))
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = require_u32_value(file, key)?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "lang_id_ecapa: `{key}` is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_u32_array(file: &GgufFile, key: &str, expected: &[usize]) -> Result<()> {
    let array = file
        .get(key)
        .and_then(GgufMetadataValue::as_array)
        .ok_or_else(|| VokraError::ModelLoad(format!("lang_id_ecapa: missing array `{key}`")))?;
    if array.element_type != GgufValueType::U32 {
        return Err(VokraError::ModelLoad(format!(
            "lang_id_ecapa: `{key}` must be a u32 array"
        )));
    }
    let actual = array
        .values
        .iter()
        .map(|value| value.as_u64().and_then(|value| usize::try_from(value).ok()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| VokraError::ModelLoad(format!("lang_id_ecapa: invalid `{key}` array")))?;
    if actual.as_slice() != expected {
        return Err(VokraError::ModelLoad(format!(
            "lang_id_ecapa: `{key}` is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_string_array(file: &GgufFile, key: &str) -> Result<Vec<String>> {
    let array = file
        .get(key)
        .and_then(GgufMetadataValue::as_array)
        .ok_or_else(|| VokraError::ModelLoad(format!("lang_id_ecapa: missing array `{key}`")))?;
    if array.element_type != GgufValueType::String {
        return Err(VokraError::ModelLoad(format!(
            "lang_id_ecapa: `{key}` must be a string array"
        )));
    }
    array
        .values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_str()
                .filter(|label| !label.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "lang_id_ecapa: `{key}` index {index} is not a non-empty string"
                    ))
                })
        })
        .collect()
}

fn require_f32(file: &GgufFile, key: &str) -> Result<f32> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(value)) if value.is_finite() => Ok(*value),
        _ => Err(VokraError::ModelLoad(format!(
            "lang_id_ecapa: missing finite f32 `{key}`"
        ))),
    }
}

fn require_exact_f32(key: &str, actual: f32, expected: f32) -> Result<()> {
    if actual.to_bits() != expected.to_bits() {
        return Err(VokraError::ModelLoad(format!(
            "lang_id_ecapa: `{key}` is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn reject_metadata(file: &GgufFile, key: &str) -> Result<()> {
    if file.get(key).is_some() {
        return Err(VokraError::ModelLoad(format!(
            "lang_id_ecapa: `{key}` is forbidden for the cosine-classifier variant"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufArray, GgufBuilder};

    fn u32_array(values: &[usize]) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: values
                .iter()
                .map(|&value| GgufMetadataValue::U32(value as u32))
                .collect(),
        })
    }

    fn labels(count: usize) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: (0..count)
                .map(|index| GgufMetadataValue::String(format!("label-{index:03}")))
                .collect(),
        })
    }

    fn contract_file(variant: LangIdVariant) -> GgufFile {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        builder.add_string(chunks::KEY_MODEL_NAME, variant.name());
        builder.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        builder.add_string(GGUF_KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf());
        builder.add_string(KEY_ARTIFACT_LAYOUT, ARTIFACT_LAYOUT);
        builder.add_string(
            KEY_UPSTREAM_REVISION,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        );
        builder.add_u32(KEY_SAMPLE_RATE, 16_000);
        builder.add_u32(KEY_N_MELS, variant.n_mels() as u32);
        builder.add_u32(KEY_TDNN_CHANNELS, SHARED_TDNN_CHANNELS as u32);
        builder.add_u32(KEY_MFA_CHANNELS, SHARED_MFA_CHANNELS as u32);
        builder.add_u32(KEY_ATTENTION_CHANNELS, SHARED_ATTENTION_CHANNELS as u32);
        builder.add_u32(KEY_RES2NET_SCALE, SHARED_RES2NET_SCALE as u32);
        builder.add_metadata(KEY_BLOCK_KERNELS, u32_array(&variant.block_kernels()));
        builder.add_metadata(KEY_BLOCK_DILATIONS, u32_array(&SHARED_DILATIONS));
        builder.add_u32(KEY_EMBEDDING_DIM, variant.embedding_dim() as u32);
        builder.add_string(KEY_CLASSIFIER_KIND, variant.classifier_kind());
        builder.add_u32(KEY_CLASS_COUNT, variant.official_class_count() as u32);
        builder.add_metadata(KEY_LABELS, labels(variant.official_class_count()));
        builder.add_f32(KEY_BN_EPS, BN_EPS);
        builder.add_f32(KEY_STATS_EPS, STATS_EPS);
        if variant == LangIdVariant::VoxLingua107 {
            builder.add_u32(KEY_CLASSIFIER_HIDDEN_DIM, 512);
            builder.add_f32(KEY_LEAKY_RELU_SLOPE, 0.01);
        }
        GgufFile::parse(builder.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn both_official_contracts_keep_distinct_topologies() {
        let vox = LangIdContract::from_gguf(&contract_file(LangIdVariant::VoxLingua107)).unwrap();
        assert_eq!(vox.variant.n_mels(), 60);
        assert_eq!(vox.variant.embedding_dim(), 256);
        assert_eq!(vox.variant.block_kernels(), [3, 3, 3]);
        assert_eq!(vox.labels.len(), 107);
        assert_eq!(vox.classifier_hidden_dim, Some(512));

        let common =
            LangIdContract::from_gguf(&contract_file(LangIdVariant::CommonLanguage)).unwrap();
        assert_eq!(common.variant.n_mels(), 80);
        assert_eq!(common.variant.embedding_dim(), 192);
        assert_eq!(common.variant.block_kernels(), [3, 3, 1]);
        assert_eq!(common.labels.len(), 45);
        assert_eq!(common.classifier_hidden_dim, None);
    }

    #[test]
    fn old_embedding_only_artifact_fails_before_forward() {
        let file = contract_file(LangIdVariant::VoxLingua107);
        let error = LangIdEcapa::from_gguf(&file).unwrap_err();
        assert!(error.to_string().contains("zero tensors"));
    }

    #[test]
    fn foreign_arch_and_cross_variant_axes_fail_closed() {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, "ecapa_tdnn");
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let error = LangIdContract::from_gguf(&file).unwrap_err();
        assert!(error.to_string().contains("expected `lang_id_ecapa`"));

        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        builder.add_string(chunks::KEY_MODEL_NAME, NAME_VOXLINGUA107);
        builder.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        builder.add_string(GGUF_KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF_VOXLINGUA107);
        builder.add_string(KEY_ARTIFACT_LAYOUT, ARTIFACT_LAYOUT);
        builder.add_string(
            KEY_UPSTREAM_REVISION,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        );
        builder.add_u32(KEY_SAMPLE_RATE, 16_000);
        builder.add_u32(KEY_N_MELS, 80);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let error = LangIdContract::from_gguf(&file).unwrap_err();
        assert!(error.to_string().contains(KEY_N_MELS));
    }

    #[test]
    fn classifier_glue_matches_probability_and_cosine_contracts() {
        let mut log_probs = vec![1.0, 2.0, -1.0];
        log_softmax_in_place(&mut log_probs).unwrap();
        let probability_sum = log_probs.iter().map(|value| value.exp()).sum::<f32>();
        assert!((probability_sum - 1.0).abs() < 1.0e-6);
        assert_eq!(
            log_probs
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .unwrap()
                .0,
            1
        );

        let mut vector = vec![3.0, 4.0];
        l2_normalize(&mut vector, L2_EPS).unwrap();
        assert!((vector[0] - 0.6).abs() < 1.0e-6);
        assert!((vector[1] - 0.8).abs() < 1.0e-6);
    }

    #[test]
    fn shared_backbone_hot_ops_are_a_subset_of_lang_id_route() {
        for op in crate::ecapa_tdnn::ECAPA_HOT_OPS {
            assert!(LANG_ID_HOT_OPS.contains(op));
        }
        assert!(LANG_ID_HOT_OPS.contains(&HotOp::Gemv));
    }
}
