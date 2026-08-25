//! Native emotion2vec+ Large speech-emotion classification on CPU and Metal.
//!
//! This module implements the exact public
//! `emotion2vec/emotion2vec_plus_large` topology: a seven-layer raw-waveform
//! encoder, five grouped relative-position convolutions, ten learned extra
//! tokens, four context blocks, eight global ALiBi blocks, mean pooling and
//! the official nine-class projection. Every learned hot operation is routed
//! through [`Compute`]; selecting any backend other than CPU or Metal fails
//! explicitly and never falls back to the CPU.
//!
//! Primary sources are the immutable official checkpoint revision
//! `6c303ba987b86b93193de93e34bb2b077a6bedc4` and FunASR source revision
//! `2f7dcbad90e82e964ab381ad63ff5109dd92327d`. The independent oracle is
//! `tools/parity/emotion2vec_dump_reference.py`.

use std::path::Path;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{CompliancePolicy, LicenseClass, Result, VokraError, check_weight_license};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

mod bound;
mod forward;
#[cfg(test)]
mod tests;

use bound::{Emotion2VecWeights, validate_metadata};
#[cfg(test)]
use forward::ForwardTaps;
use forward::run_forward;

/// Converter/runtime architecture handshake.
pub const ARCH: &str = "emotion2vec";
/// Canonical public GGUF model name.
pub const NAME: &str = "emotion2vec-plus-large";
/// Model-zoo task category.
pub const CATEGORY: &str = "emotion";
/// Immutable official checkpoint repository.
pub const UPSTREAM_HF: &str = "emotion2vec/emotion2vec_plus_large";
/// Immutable official checkpoint revision.
pub const UPSTREAM_REVISION: &str = "6c303ba987b86b93193de93e34bb2b077a6bedc4";
/// Official checkpoint filename.
pub const CHECKPOINT_FILE: &str = "model.pt";
/// Official checkpoint SHA-256.
pub const CHECKPOINT_SHA256: &str =
    "be501a01f26fcdc7663a062dff86af839afbaef7c4de32f5e42d7e1ad2784da4";
/// Immutable FunASR source revision used by the parity oracle.
pub const FUNASR_REVISION: &str = "2f7dcbad90e82e964ab381ad63ff5109dd92327d";

/// Required mono PCM sample rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// Transformer residual width.
pub const HIDDEN: usize = 1_024;
/// Global emotion2vec block count.
pub const GLOBAL_LAYERS: usize = 8;
/// Audio-context block count.
pub const CONTEXT_LAYERS: usize = 4;
/// Attention head count.
pub const HEADS: usize = 16;
/// Feed-forward inner width.
pub const FFN: usize = 4_096;
/// Learned prefix-token count.
pub const EXTRA_TOKENS: usize = 10;
/// Official emotion-class count.
pub const NUM_CLASSES: usize = 9;
/// Grouped positional-convolution depth.
pub const POSITION_LAYERS: usize = 5;
/// Grouped positional-convolution kernel width.
pub const POSITION_KERNEL: usize = 19;
/// Grouped positional-convolution group count.
pub const POSITION_GROUPS: usize = 16;
/// LayerNorm epsilon used by the official implementation.
pub const LAYER_NORM_EPS: f32 = 1.0e-5;

pub(super) const FEATURE_DIM: usize = 512;
pub(super) const TENSOR_COUNT: usize = 185;
pub(super) const CONV_DIM: [usize; 7] = [FEATURE_DIM; 7];
pub(super) const CONV_KERNEL: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
pub(super) const CONV_STRIDE: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];
pub(super) const MIN_PCM_SAMPLES: usize = 400;
pub(super) const PREFIX: &str = "vokra.emotion2vec";

/// Official bilingual class labels in classifier-row order.
pub const EMOTION_CLASS_LABELS: [&str; NUM_CLASSES] = [
    "生气/angry",
    "厌恶/disgusted",
    "恐惧/fearful",
    "开心/happy",
    "中立/neutral",
    "其他/other",
    "难过/sad",
    "吃惊/surprised",
    "<unk>",
];

/// Every learned operation required by the CPU/Metal forward.
pub const EMOTION2VEC_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
];

pub(super) const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: "emotion2vec-plus-large",
    arch: ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: TENSOR_COUNT,
    manifest_sha256: [
        0xf5, 0xf8, 0xf6, 0x84, 0x30, 0x2c, 0xf5, 0x5f, 0xb3, 0x99, 0x27, 0x7a, 0x74, 0x46, 0x97,
        0x6a, 0x77, 0xf5, 0x70, 0x81, 0x6e, 0x7e, 0x33, 0x45, 0xa0, 0x08, 0xe4, 0xd0, 0xb6, 0x77,
        0x44, 0x01,
    ],
};

/// Fixed topology resolved from the audited checkpoint contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Emotion2VecConfig {
    /// Required input sample rate.
    pub sample_rate: u32,
    /// Transformer residual width.
    pub hidden_size: usize,
    /// Audio-context block count.
    pub context_layers: usize,
    /// Global block count.
    pub global_layers: usize,
    /// Attention-head count.
    pub num_heads: usize,
    /// Feed-forward inner width.
    pub intermediate_size: usize,
    /// Learned prefix-token count.
    pub num_extra_tokens: usize,
    /// LayerNorm epsilon.
    pub layer_norm_eps: f32,
}

impl Default for Emotion2VecConfig {
    fn default() -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            hidden_size: HIDDEN,
            context_layers: CONTEXT_LAYERS,
            global_layers: GLOBAL_LAYERS,
            num_heads: HEADS,
            intermediate_size: FFN,
            num_extra_tokens: EXTRA_TOKENS,
            layer_norm_eps: LAYER_NORM_EPS,
        }
    }
}

/// Native emotion2vec+ Large classifier.
#[derive(Debug)]
pub struct Emotion2Vec {
    checkpoint: StrictCheckpoint,
    config: Emotion2VecConfig,
    weights: Emotion2VecWeights,
    backend: BackendKind,
    legacy_metadata_repaired: bool,
}

impl Emotion2Vec {
    /// Opens and strictly binds the canonical public GGUF.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = GgufFile::open(path)?;
        Self::from_gguf(&file)
    }

    /// Strictly binds an already-open GGUF under the default commercial policy.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Self::from_gguf_with_policy(file, &CompliancePolicy::strict())
    }

    /// Strictly binds an already-open GGUF under an explicit compliance policy.
    pub fn from_gguf_with_policy(file: &GgufFile, policy: &CompliancePolicy) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        let license = check_weight_license(file, policy)?;
        if license.class != LicenseClass::Permissive
            || checkpoint.weight_license() != LicenseClass::Permissive
        {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: weight license resolves to {}, expected Permissive for the owner-signed MIT release",
                license.class.as_str()
            )));
        }
        let legacy_metadata_repaired = validate_metadata(file)?;
        let weights = Emotion2VecWeights::bind(file)?;
        Ok(Self {
            checkpoint,
            config: Emotion2VecConfig::default(),
            weights,
            backend: BackendKind::Cpu,
            legacy_metadata_repaired,
        })
    }

    /// Selects the execution backend. Only CPU and Metal are supported.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the selected execution backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the audited fixed topology.
    #[must_use]
    pub const fn config(&self) -> &Emotion2VecConfig {
        &self.config
    }

    /// Returns the canonical bound model name.
    #[must_use]
    pub fn model_name(&self) -> &str {
        self.checkpoint.model_name()
    }

    /// Returns the exact bound tensor count.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.checkpoint.tensor_count()
    }

    /// Returns the owner-signed weight-license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    /// Whether the historical public GGUF omitted the additive topology group.
    #[must_use]
    pub const fn legacy_metadata_repaired(&self) -> bool {
        self.legacy_metadata_repaired
    }

    /// Returns the official bilingual labels in classifier-row order.
    #[must_use]
    pub const fn class_labels() -> &'static [&'static str; NUM_CLASSES] {
        &EMOTION_CLASS_LABELS
    }

    /// Runs the encoder and returns frame-major final features without prefix tokens.
    pub fn encode_features(&self, pcm: &[f32], sample_rate: u32) -> Result<(Vec<f32>, usize)> {
        validate_pcm(pcm, sample_rate)?;
        let compute = self.compute()?;
        let result = run_forward(&self.weights, pcm, &compute, None, false)?;
        Ok((result.final_features, result.frames))
    }

    /// Returns raw nine-way logits for mono 16 kHz PCM.
    pub fn classify_pcm(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        validate_pcm(pcm, sample_rate)?;
        let compute = self.compute()?;
        Ok(run_forward(&self.weights, pcm, &compute, None, true)?.logits)
    }

    /// Preserves the original API: raw nine-way logits for implicit 16 kHz PCM.
    pub fn classify(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        self.classify_pcm(pcm, SAMPLE_RATE)
    }

    /// Returns softmax probabilities in [`EMOTION_CLASS_LABELS`] order.
    pub fn classify_scores(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        validate_pcm(pcm, sample_rate)?;
        let compute = self.compute()?;
        Ok(run_forward(&self.weights, pcm, &compute, None, true)?.scores)
    }

    fn compute(&self) -> Result<Compute> {
        match self.backend {
            BackendKind::Cpu | BackendKind::Metal => {
                Compute::for_backend(self.backend, EMOTION2VEC_HOT_OPS)
            }
            other => Err(VokraError::UnsupportedOp(format!(
                "emotion2vec: backend {other:?} is unsupported; this model wave implements only Mac CPU and Metal. Vokra will not silently execute any learned operation on the CPU (FR-EX-08)"
            ))),
        }
    }

    #[cfg(test)]
    fn classify_with_taps(
        &self,
        pcm: &[f32],
        sample_rate: u32,
    ) -> Result<(Vec<f32>, Vec<f32>, ForwardTaps)> {
        validate_pcm(pcm, sample_rate)?;
        let compute = self.compute()?;
        let mut taps = ForwardTaps::default();
        let result = run_forward(&self.weights, pcm, &compute, Some(&mut taps), true)?;
        Ok((result.logits, result.scores, taps))
    }
}

fn validate_pcm(pcm: &[f32], sample_rate: u32) -> Result<()> {
    if sample_rate != SAMPLE_RATE {
        return Err(VokraError::InvalidArgument(format!(
            "emotion2vec: expected {SAMPLE_RATE} Hz mono PCM, got {sample_rate} Hz; resampling is caller-owned"
        )));
    }
    if pcm.len() < MIN_PCM_SAMPLES {
        return Err(VokraError::InvalidArgument(format!(
            "emotion2vec: PCM has {} samples, but the seven-layer waveform stem requires at least {MIN_PCM_SAMPLES}",
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
            "emotion2vec: PCM sample {index} is non-finite ({value})"
        )));
    }
    Ok(())
}
