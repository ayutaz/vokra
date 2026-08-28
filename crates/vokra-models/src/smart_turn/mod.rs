//! Native smart-turn v2 semantic endpointing.
//!
//! `pipecat-ai/smart-turn-v2` is a raw-waveform Wav2Vec2-base encoder with
//! learned attention pooling and a binary classifier. It produces one
//! completion probability for an utterance; it is not a frame-level VAD and
//! therefore deliberately does not implement `VadEngine`.
//!
//! The binder accepts only the converter's pinned 221-tensor canonical GGUF.
//! Inference reproduces the official 16-second right-padding masks without
//! allocating or evaluating all padded queries: valid encoder keys plus the
//! small zero-query suffix retained by Pipecat's ratio-based pooling mask are
//! sufficient for an identical eval-mode result.

use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, chunks};
use vokra_core::{BackendKind, LicenseClass, Result, VokraError};
use vokra_ops::{
    ConvLayerWeights, WaveformFrontendAttrs, WaveformFrontendWeights,
    waveform_frontend_with_right_padding,
};

use crate::align::charsiu::{
    CharsiuBlock, CharsiuConfig, CharsiuFeatureProjection, CharsiuPosConv,
    feature_projection_forward, feature_projection_forward_with_compute, gelu_exact,
    layer_norm_inplace, layer_norm_with_compute_inplace, linear_forward,
    linear_forward_with_compute, positional_conv_forward, positional_conv_forward_with_compute,
    transformer_block_forward_with_valid_keys,
    transformer_block_forward_with_valid_keys_and_compute,
};
use crate::compute::{Compute, HotOp};

/// Canonical GGUF architecture tag.
pub const ARCH: &str = "smart_turn";
/// Canonical model name stored in GGUF metadata.
pub const NAME: &str = "smart-turn-v2";
/// Catalog category used by the upstream model card.
pub const CATEGORY: &str = "vad";
/// Pinned upstream Hugging Face repository.
pub const UPSTREAM_HF: &str = "pipecat-ai/smart-turn-v2";
/// Default SPDX identifier for the released weights.
pub const DEFAULT_LICENSE_SPDX: &str = "bsd-2-clause";

/// Primary checkpoint source used by the converter.
pub const PRIMARY_SOURCE_HF: &str = "huggingface.co/pipecat-ai/smart-turn-v2";
/// Pinned Pipecat endpoint-head implementation.
pub const PRIMARY_SOURCE_CODE: &str = "github.com/pipecat-ai/pipecat/blob/c560a748b4213ca8db6f43a5d165d91aaa124a52/src/pipecat/audio/turn/smart_turn/local_smart_turn_v2.py";
/// Upstream Wav2Vec2 backbone source.
pub const PRIMARY_SOURCE_BACKBONE_HF: &str = "huggingface.co/facebook/wav2vec2-base-960h";
/// Wav2Vec2 architecture paper.
pub const PRIMARY_SOURCE_BACKBONE_PAPER: &str = "arxiv.org/abs/2006.11477";
/// Independent Python reference fixture generator.
pub const SIDECAR_PATH: &str = "tools/parity/smart_turn_prepare_checkpoint.py";

/// GGUF metadata key for the catalog category.
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// GGUF metadata key for the upstream repository.
pub const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
/// GGUF metadata key for the expected PCM sample rate.
pub const KEY_SMART_TURN_SAMPLE_RATE: &str = "vokra.smart_turn.sample_rate";
/// GGUF metadata key for the maximum utterance duration.
pub const KEY_SMART_TURN_MAX_SEGMENT_SECONDS: &str = "vokra.smart_turn.max_segment_seconds";
/// Metadata keys that must be present or absent as one group.
pub const SEGMENT_SPEC_KEYS: [&str; 2] = [
    KEY_SMART_TURN_SAMPLE_RATE,
    KEY_SMART_TURN_MAX_SEGMENT_SECONDS,
];

/// Official input sample rate.
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 16_000;
/// Official fixed processor window in seconds.
pub const GUARD_MAX_SEGMENT_SECONDS: f32 = 16.0;
/// Default threshold for a completed-turn decision.
pub const DEFAULT_COMPLETION_THRESHOLD: f32 = 0.5;

const REVISION: &str = "3267e96b50db03fe030b9869eb35f849a5eea1fa";
const CHECKPOINT_SHA256: &str = "0c4429a3f55d42d055e08903eb961f6ec4021c9e35d489007f3dc4981b6b028b";
const CONFIG_SHA256: &str = "31aa20aebdee3f961077a9482f909efce4d46199aabd848def1c4d9456e2c716";
const PREPROCESSOR_CONFIG_SHA256: &str =
    "617bd0950f8cc9ac4062e8c73a7be60305ca5790a243df55fa6f44fb671b55b1";
const REFERENCE_REVISION: &str = "c560a748b4213ca8db6f43a5d165d91aaa124a52";
const HIDDEN: usize = 768;
const FEATURE_DIM: usize = 512;
const FFN: usize = 3072;
const N_LAYER: usize = 12;
const N_HEAD: usize = 12;
const POS_KERNEL: usize = 128;
const POS_GROUPS: usize = 16;
const MAX_INPUT_SAMPLES: usize = 256_000;
const NORMALIZATION_EPS: f32 = 1e-7;
const TENSOR_COUNT: usize = 221;

/// Complete learned-op registry for the SmartTurn Metal route.
pub const SMART_TURN_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
];

#[derive(Debug, Clone, Copy, PartialEq)]
/// One validated utterance-level completion prediction.
pub struct TurnPrediction {
    completion_probability: f32,
}

impl TurnPrediction {
    /// Validates and constructs a probability in the closed interval `[0, 1]`.
    pub fn new(probability: f32) -> Result<Self> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(VokraError::InvalidArgument(format!(
                "smart_turn: completion probability {probability} is not finite in [0, 1]"
            )));
        }
        Ok(Self {
            completion_probability: probability,
        })
    }

    #[must_use]
    /// Returns the predicted probability that the speaker completed the turn.
    pub const fn completion_probability(&self) -> f32 {
        self.completion_probability
    }

    #[must_use]
    /// Applies a caller-selected completion threshold.
    pub fn is_complete(&self, threshold: f32) -> bool {
        self.completion_probability >= threshold
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Optional processor geometry transcribed into GGUF metadata.
pub struct SmartTurnSegmentSpec {
    /// Required PCM sample rate.
    pub sample_rate: u32,
    /// Maximum accepted utterance duration in seconds.
    pub max_segment_seconds: f32,
}

impl SmartTurnSegmentSpec {
    /// Validates the sample-rate and duration fields.
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(VokraError::ModelLoad(format!(
                "smart_turn: `{KEY_SMART_TURN_SAMPLE_RATE}` must be positive"
            )));
        }
        if !self.max_segment_seconds.is_finite() || self.max_segment_seconds <= 0.0 {
            return Err(VokraError::ModelLoad(format!(
                "smart_turn: `{KEY_SMART_TURN_MAX_SEGMENT_SECONDS}` must be finite and positive"
            )));
        }
        Ok(())
    }

    /// Reads the all-or-nothing segment metadata group from a GGUF.
    pub fn from_gguf(file: &GgufFile) -> Result<Option<Self>> {
        let any = SEGMENT_SPEC_KEYS.iter().any(|key| file.get(key).is_some());
        if !any {
            return Ok(None);
        }
        let spec = Self {
            sample_rate: meta_u32(file, KEY_SMART_TURN_SAMPLE_RATE)?,
            max_segment_seconds: meta_f32(file, KEY_SMART_TURN_MAX_SEGMENT_SECONDS)?,
        };
        spec.validate()?;
        Ok(Some(spec))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Runtime configuration loaded from SmartTurn GGUF metadata.
pub struct SmartTurnConfig {
    /// Explicit segment geometry, or `None` for legacy canonical defaults.
    pub segment: Option<SmartTurnSegmentSpec>,
}

impl SmartTurnConfig {
    /// Loads and validates SmartTurn runtime metadata.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Ok(Self {
            segment: SmartTurnSegmentSpec::from_gguf(file)?,
        })
    }

    #[must_use]
    /// Returns the explicit sample rate or the pinned upstream default.
    pub fn expected_sample_rate(&self) -> u32 {
        self.segment
            .map(|value| value.sample_rate)
            .unwrap_or(DEFAULT_SAMPLE_RATE_HZ)
    }

    #[must_use]
    /// Reports whether the sample rate came from the canonical default.
    pub fn sample_rate_is_assumed(&self) -> bool {
        self.segment.is_none()
    }

    #[must_use]
    /// Returns the explicit maximum duration or the pinned upstream default.
    pub fn max_segment_seconds(&self) -> f32 {
        self.segment
            .map(|value| value.max_segment_seconds)
            .unwrap_or(GUARD_MAX_SEGMENT_SECONDS)
    }

    #[must_use]
    /// Converts the duration guard to a sample count for `sample_rate`.
    pub fn max_segment_samples(&self, sample_rate: u32) -> usize {
        (f64::from(self.max_segment_seconds()) * f64::from(sample_rate)).floor() as usize
    }
}

#[derive(Debug)]
struct EndpointHead {
    pool_w1: Vec<f32>,
    pool_b1: Vec<f32>,
    pool_w2: Vec<f32>,
    pool_b2: Vec<f32>,
    classifier_w1: Vec<f32>,
    classifier_b1: Vec<f32>,
    classifier_norm_gamma: Vec<f32>,
    classifier_norm_beta: Vec<f32>,
    classifier_w2: Vec<f32>,
    classifier_b2: Vec<f32>,
    classifier_w3: Vec<f32>,
    classifier_b3: Vec<f32>,
}

#[derive(Debug)]
/// Strictly bound canonical SmartTurn tensor bundle.
pub struct SmartTurnWeights {
    tensors: Vec<(String, Vec<usize>)>,
    stem_attrs: WaveformFrontendAttrs,
    stem_weights: WaveformFrontendWeights,
    feature_projection: CharsiuFeatureProjection,
    pos_conv: CharsiuPosConv,
    encoder_norm_gamma: Vec<f32>,
    encoder_norm_beta: Vec<f32>,
    blocks: Vec<CharsiuBlock>,
    head: EndpointHead,
}

impl SmartTurnWeights {
    /// Binds all 221 canonical tensors and rejects missing or foreign shapes.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        if file.tensors().len() != TENSOR_COUNT {
            return Err(VokraError::ModelLoad(format!(
                "smart_turn GGUF has {} tensors, expected exactly {TENSOR_COUNT}",
                file.tensors().len()
            )));
        }
        let tensors = file
            .tensors()
            .iter()
            .map(|info| {
                (
                    info.name.clone(),
                    info.dimensions.iter().map(|&dim| dim as usize).collect(),
                )
            })
            .collect();

        let stem_attrs = WaveformFrontendAttrs::wav2vec2_base();
        let mut stem_layers = Vec::with_capacity(stem_attrs.layers.len());
        let mut in_channels = 1usize;
        for (i, attrs) in stem_attrs.layers.iter().enumerate() {
            let prefix = format!("wav2vec2.feature_extractor.conv_layers.{i}");
            let (norm_gamma, norm_beta) = if i == 0 {
                (
                    Some(tensor(
                        file,
                        &format!("{prefix}.layer_norm.weight"),
                        &[FEATURE_DIM],
                    )?),
                    Some(tensor(
                        file,
                        &format!("{prefix}.layer_norm.bias"),
                        &[FEATURE_DIM],
                    )?),
                )
            } else {
                (None, None)
            };
            stem_layers.push(ConvLayerWeights {
                conv_w: tensor(
                    file,
                    &format!("{prefix}.conv.weight"),
                    &[FEATURE_DIM, in_channels, attrs.kernel],
                )?,
                conv_b: Vec::new(),
                norm_gamma,
                norm_beta,
            });
            in_channels = FEATURE_DIM;
        }

        let feature_projection = CharsiuFeatureProjection {
            norm_gamma: Some(tensor(
                file,
                "wav2vec2.feature_projection.layer_norm.weight",
                &[FEATURE_DIM],
            )?),
            norm_beta: Some(tensor(
                file,
                "wav2vec2.feature_projection.layer_norm.bias",
                &[FEATURE_DIM],
            )?),
            linear_w: tensor(
                file,
                "wav2vec2.feature_projection.projection.weight",
                &[HIDDEN, FEATURE_DIM],
            )?,
            linear_b: tensor(
                file,
                "wav2vec2.feature_projection.projection.bias",
                &[HIDDEN],
            )?,
        };
        let pos_conv = CharsiuPosConv {
            weight: tensor(
                file,
                "smart_turn.pos_conv.weight",
                &[HIDDEN, HIDDEN / POS_GROUPS, POS_KERNEL],
            )?,
            bias: tensor(file, "smart_turn.pos_conv.bias", &[HIDDEN])?,
        };
        let encoder_norm_gamma = tensor(file, "wav2vec2.encoder.layer_norm.weight", &[HIDDEN])?;
        let encoder_norm_beta = tensor(file, "wav2vec2.encoder.layer_norm.bias", &[HIDDEN])?;

        let mut blocks = Vec::with_capacity(N_LAYER);
        for i in 0..N_LAYER {
            let p = format!("wav2vec2.encoder.layers.{i}");
            blocks.push(CharsiuBlock {
                attn_norm_gamma: tensor(file, &format!("{p}.layer_norm.weight"), &[HIDDEN])?,
                attn_norm_beta: tensor(file, &format!("{p}.layer_norm.bias"), &[HIDDEN])?,
                q_w: tensor(
                    file,
                    &format!("{p}.attention.q_proj.weight"),
                    &[HIDDEN, HIDDEN],
                )?,
                q_b: tensor(file, &format!("{p}.attention.q_proj.bias"), &[HIDDEN])?,
                k_w: tensor(
                    file,
                    &format!("{p}.attention.k_proj.weight"),
                    &[HIDDEN, HIDDEN],
                )?,
                k_b: tensor(file, &format!("{p}.attention.k_proj.bias"), &[HIDDEN])?,
                v_w: tensor(
                    file,
                    &format!("{p}.attention.v_proj.weight"),
                    &[HIDDEN, HIDDEN],
                )?,
                v_b: tensor(file, &format!("{p}.attention.v_proj.bias"), &[HIDDEN])?,
                o_w: tensor(
                    file,
                    &format!("{p}.attention.out_proj.weight"),
                    &[HIDDEN, HIDDEN],
                )?,
                o_b: tensor(file, &format!("{p}.attention.out_proj.bias"), &[HIDDEN])?,
                ffn_norm_gamma: tensor(file, &format!("{p}.final_layer_norm.weight"), &[HIDDEN])?,
                ffn_norm_beta: tensor(file, &format!("{p}.final_layer_norm.bias"), &[HIDDEN])?,
                fc1_w: tensor(
                    file,
                    &format!("{p}.feed_forward.intermediate_dense.weight"),
                    &[FFN, HIDDEN],
                )?,
                fc1_b: tensor(
                    file,
                    &format!("{p}.feed_forward.intermediate_dense.bias"),
                    &[FFN],
                )?,
                fc2_w: tensor(
                    file,
                    &format!("{p}.feed_forward.output_dense.weight"),
                    &[HIDDEN, FFN],
                )?,
                fc2_b: tensor(
                    file,
                    &format!("{p}.feed_forward.output_dense.bias"),
                    &[HIDDEN],
                )?,
            });
        }
        let head = EndpointHead {
            pool_w1: tensor(file, "pool_attention.0.weight", &[256, HIDDEN])?,
            pool_b1: tensor(file, "pool_attention.0.bias", &[256])?,
            pool_w2: tensor(file, "pool_attention.2.weight", &[1, 256])?,
            pool_b2: tensor(file, "pool_attention.2.bias", &[1])?,
            classifier_w1: tensor(file, "classifier.0.weight", &[256, HIDDEN])?,
            classifier_b1: tensor(file, "classifier.0.bias", &[256])?,
            classifier_norm_gamma: tensor(file, "classifier.1.weight", &[256])?,
            classifier_norm_beta: tensor(file, "classifier.1.bias", &[256])?,
            classifier_w2: tensor(file, "classifier.4.weight", &[64, 256])?,
            classifier_b2: tensor(file, "classifier.4.bias", &[64])?,
            classifier_w3: tensor(file, "classifier.6.weight", &[1, 64])?,
            classifier_b3: tensor(file, "classifier.6.bias", &[1])?,
        };
        Ok(Self {
            tensors,
            stem_attrs,
            stem_weights: WaveformFrontendWeights {
                layers: stem_layers,
            },
            feature_projection,
            pos_conv,
            encoder_norm_gamma,
            encoder_norm_beta,
            blocks,
            head,
        })
    }

    #[must_use]
    /// Returns the number of tensors consumed by the binder.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    #[must_use]
    /// Returns the bound tensor-name and shape manifest.
    pub fn tensors(&self) -> &[(String, Vec<usize>)] {
        &self.tensors
    }
}

#[derive(Debug)]
/// Native SmartTurn v2 utterance-level endpoint classifier.
pub struct SmartTurn {
    cfg: SmartTurnConfig,
    weights: SmartTurnWeights,
    weight_license: LicenseClass,
    model_name: Option<String>,
    category: Option<String>,
    upstream_hf: Option<String>,
    backend: BackendKind,
}

impl SmartTurn {
    /// Binds a parsed canonical SmartTurn GGUF.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        require_string(file, "vokra.smart_turn.revision", REVISION)?;
        require_string(
            file,
            "vokra.smart_turn.checkpoint_sha256",
            CHECKPOINT_SHA256,
        )?;
        require_string(file, "vokra.smart_turn.config_sha256", CONFIG_SHA256)?;
        require_string(
            file,
            "vokra.smart_turn.preprocessor_config_sha256",
            PREPROCESSOR_CONFIG_SHA256,
        )?;
        require_string(
            file,
            "vokra.smart_turn.reference_revision",
            REFERENCE_REVISION,
        )?;
        for (key, expected) in [
            ("vokra.smart_turn.hidden_size", HIDDEN as u32),
            ("vokra.smart_turn.feature_dim", FEATURE_DIM as u32),
            ("vokra.smart_turn.ffn_dim", FFN as u32),
            ("vokra.smart_turn.n_layer", N_LAYER as u32),
            ("vokra.smart_turn.n_head", N_HEAD as u32),
            ("vokra.smart_turn.pos_conv_kernel", POS_KERNEL as u32),
            ("vokra.smart_turn.pos_conv_groups", POS_GROUPS as u32),
            (
                "vokra.smart_turn.max_input_samples",
                MAX_INPUT_SAMPLES as u32,
            ),
        ] {
            let actual = meta_u32(file, key)?;
            if actual != expected {
                return Err(VokraError::ModelLoad(format!(
                    "smart_turn GGUF `{key}` is {actual}, expected {expected}"
                )));
            }
        }
        for (key, expected) in [
            ("vokra.smart_turn.layer_norm_eps", 1e-5f32),
            ("vokra.smart_turn.normalization_eps", NORMALIZATION_EPS),
            (
                "vokra.smart_turn.completion_threshold",
                DEFAULT_COMPLETION_THRESHOLD,
            ),
        ] {
            let actual = meta_f32(file, key)?;
            if actual.to_bits() != expected.to_bits() {
                return Err(VokraError::ModelLoad(format!(
                    "smart_turn GGUF `{key}` is {actual}, expected {expected}"
                )));
            }
        }
        let cfg = SmartTurnConfig::from_gguf(file)?;
        let segment = cfg.segment.ok_or_else(|| {
            VokraError::ModelLoad(
                "smart_turn GGUF is missing its required segment contract".to_owned(),
            )
        })?;
        if segment.sample_rate != DEFAULT_SAMPLE_RATE_HZ
            || segment.max_segment_seconds.to_bits() != 16.0f32.to_bits()
        {
            return Err(VokraError::ModelLoad(format!(
                "smart_turn GGUF segment contract is {segment:?}, expected 16000 Hz / 16 s"
            )));
        }
        let weights = SmartTurnWeights::from_gguf(file)?;
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(GgufMetadataValue::as_str)
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        Ok(Self {
            cfg,
            weights,
            weight_license,
            model_name: optional_string(file, chunks::KEY_MODEL_NAME),
            category: optional_string(file, KEY_MODEL_CATEGORY),
            upstream_hf: optional_string(file, KEY_PROVENANCE_UPSTREAM_HF),
            backend: BackendKind::Cpu,
        })
    }

    /// Reads and binds a canonical SmartTurn GGUF from disk.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let file = GgufFile::open(path)?;
        Self::from_gguf(&file)
    }

    #[must_use]
    /// Returns the validated runtime configuration.
    pub const fn config(&self) -> &SmartTurnConfig {
        &self.cfg
    }

    /// Selects the backend for the complete learned SmartTurn forward.
    /// CPU remains the exact established oracle; non-CPU execution is gated
    /// by [`SMART_TURN_HOT_OPS`] as one indivisible model route.
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

    #[must_use]
    /// Returns the number of bound tensors.
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    #[must_use]
    /// Returns the bound tensor-name and shape manifest.
    pub fn tensors(&self) -> &[(String, Vec<usize>)] {
        self.weights.tensors()
    }

    #[must_use]
    /// Returns the optional GGUF model name.
    pub fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    #[must_use]
    /// Returns the optional GGUF catalog category.
    pub fn category(&self) -> Option<&str> {
        self.category.as_deref()
    }

    #[must_use]
    /// Returns the optional upstream repository provenance value.
    pub fn upstream_hf(&self) -> Option<&str> {
        self.upstream_hf.as_deref()
    }

    #[must_use]
    /// Returns the audited weight-license class.
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    #[must_use]
    /// Reports whether the loaded weight license is research-only.
    pub fn is_research_only(&self) -> bool {
        self.weight_license.requires_research_flag()
    }

    /// Predicts one completion probability for a complete mono utterance.
    pub fn predict_endpoint(&self, pcm: &[f32], sample_rate: u32) -> Result<TurnPrediction> {
        if sample_rate != self.cfg.expected_sample_rate() {
            return Err(VokraError::InvalidArgument(format!(
                "smart_turn: sample rate {sample_rate} does not match required {} Hz; no silent resampling",
                self.cfg.expected_sample_rate()
            )));
        }
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "smart_turn: utterance PCM is empty".to_owned(),
            ));
        }
        if pcm.len() > MAX_INPUT_SAMPLES {
            return Err(VokraError::InvalidArgument(format!(
                "smart_turn: {} input samples exceed the canonical {MAX_INPUT_SAMPLES}-sample (16 s) window",
                pcm.len()
            )));
        }
        if pcm.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "smart_turn: PCM contains a non-finite sample".to_owned(),
            ));
        }

        // The first Wav2Vec2 convolution uses GroupNorm across the complete
        // time axis. Pipecat always right-pads to 16 seconds before the
        // encoder, so the padded convolution frames affect even the valid
        // prefix. Preserve that contract here; only the Transformer queries
        // can be trimmed safely.
        let normalized = zero_mean_unit_var(pcm, NORMALIZATION_EPS);
        let compute = if self.backend == BackendKind::Cpu {
            None
        } else {
            Some(Compute::for_backend(self.backend, SMART_TURN_HOT_OPS)?)
        };
        self.predict_endpoint_validated(pcm.len(), &normalized, compute.as_ref())
    }

    fn predict_endpoint_validated(
        &self,
        pcm_len: usize,
        normalized: &[f32],
        compute: Option<&Compute>,
    ) -> Result<TurnPrediction> {
        let features = if let Some(compute) = compute {
            waveform_frontend_with_compute(
                normalized,
                MAX_INPUT_SAMPLES,
                &self.weights.stem_attrs,
                &self.weights.stem_weights,
                compute,
            )?
        } else {
            waveform_frontend_with_right_padding(
                normalized,
                MAX_INPUT_SAMPLES,
                &self.weights.stem_attrs,
                &self.weights.stem_weights,
            )?
        };
        let valid_frames = features.len() / FEATURE_DIM;
        let full_frames = self.weights.stem_attrs.predict_t_out(MAX_INPUT_SAMPLES)?;
        let pooled_frames = ratio_mask_frames(pcm_len, MAX_INPUT_SAMPLES, full_frames);
        if pooled_frames < valid_frames || pooled_frames > full_frames {
            return Err(VokraError::ModelLoad(format!(
                "smart_turn: invalid mask geometry: valid={valid_frames}, pooled={pooled_frames}, full={full_frames}"
            )));
        }

        let encoder_cfg = CharsiuConfig::default_charsiu_en();
        let mut hidden = if let Some(compute) = compute.as_ref() {
            feature_projection_forward_with_compute(
                &features[..valid_frames * FEATURE_DIM],
                valid_frames,
                FEATURE_DIM,
                &self.weights.feature_projection,
                HIDDEN,
                true,
                1e-5,
                compute,
            )?
        } else {
            feature_projection_forward(
                &features[..valid_frames * FEATURE_DIM],
                valid_frames,
                FEATURE_DIM,
                &self.weights.feature_projection,
                HIDDEN,
                true,
                1e-5,
            )
        };
        hidden.resize(pooled_frames * HIDDEN, 0.0);
        let position = if let Some(compute) = compute.as_ref() {
            positional_conv_forward_with_compute(
                &hidden,
                pooled_frames,
                &encoder_cfg,
                &self.weights.pos_conv,
                compute,
            )?
        } else {
            positional_conv_forward(&hidden, pooled_frames, &encoder_cfg, &self.weights.pos_conv)?
        };
        for (value, positional) in hidden.iter_mut().zip(position) {
            *value += positional;
        }
        if let Some(compute) = compute.as_ref() {
            layer_norm_with_compute_inplace(
                &mut hidden,
                pooled_frames,
                HIDDEN,
                &self.weights.encoder_norm_gamma,
                &self.weights.encoder_norm_beta,
                1e-5,
                compute,
            )?;
        } else {
            layer_norm_inplace(
                &mut hidden,
                pooled_frames,
                HIDDEN,
                &self.weights.encoder_norm_gamma,
                &self.weights.encoder_norm_beta,
                1e-5,
            );
        }
        for block in &self.weights.blocks {
            if let Some(compute) = compute.as_ref() {
                transformer_block_forward_with_valid_keys_and_compute(
                    &mut hidden,
                    pooled_frames,
                    valid_frames,
                    &encoder_cfg,
                    block,
                    compute,
                )?;
            } else {
                transformer_block_forward_with_valid_keys(
                    &mut hidden,
                    pooled_frames,
                    valid_frames,
                    &encoder_cfg,
                    block,
                );
            }
        }
        let probability = if let Some(compute) = compute.as_ref() {
            endpoint_head_with_compute(&hidden, pooled_frames, &self.weights.head, compute)?
        } else {
            endpoint_head(&hidden, pooled_frames, &self.weights.head)
        };
        TurnPrediction::new(probability)
    }
}

/// Backend-dispatched wav2vec2 stem for SmartTurn's fixed right-padded input
/// contract. Convolutions and GELU execute on the selected backend. The first
/// per-channel GroupNorm statistics are host reduction/glue over the padded
/// convolution result; no learned convolution is substituted with a CPU path.
fn waveform_frontend_with_compute(
    waveform: &[f32],
    padded_time: usize,
    attrs: &WaveformFrontendAttrs,
    weights: &WaveformFrontendWeights,
    compute: &Compute,
) -> Result<Vec<f32>> {
    weights.validate(attrs)?;
    if waveform.len() % attrs.in_channels != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "smart_turn waveform length {} is not divisible by {} channel(s)",
            waveform.len(),
            attrs.in_channels
        )));
    }
    let mut valid_time = waveform.len() / attrs.in_channels;
    if padded_time < valid_time {
        return Err(VokraError::InvalidArgument(format!(
            "smart_turn padded time {padded_time} is shorter than input {valid_time}"
        )));
    }
    let _ = attrs.predict_t_out(valid_time)?;
    let mut current = waveform.to_vec();
    let mut in_channels = attrs.in_channels;

    for (index, (layer, layer_weights)) in attrs.layers.iter().zip(&weights.layers).enumerate() {
        let use_padded_axis = index == 0 && attrs.norm.has_group_norm(index);
        let convolution_time = if use_padded_axis {
            padded_time
        } else {
            valid_time
        };
        let mut convolution_input = vec![0.0f32; in_channels * convolution_time];
        for channel in 0..in_channels {
            let source = &current[channel * valid_time..(channel + 1) * valid_time];
            let destination = &mut convolution_input
                [channel * convolution_time..channel * convolution_time + valid_time];
            destination.copy_from_slice(source);
        }
        let full_out_time = (convolution_time - layer.kernel) / layer.stride + 1;
        let valid_out_time = (valid_time - layer.kernel) / layer.stride + 1;
        let mut convolution = vec![0.0f32; layer.out_channels * full_out_time];
        compute.conv1d_f32(
            &convolution_input,
            in_channels,
            convolution_time,
            &layer_weights.conv_w,
            layer.out_channels,
            layer.kernel,
            attrs.conv_bias.then_some(layer_weights.conv_b.as_slice()),
            layer.stride,
            0,
            &mut convolution,
        )?;

        if attrs.norm.has_group_norm(index) {
            let gamma = layer_weights.norm_gamma.as_ref().expect("validated gamma");
            let beta = layer_weights.norm_beta.as_ref().expect("validated beta");
            group_norm_each_channel(
                &mut convolution,
                layer.out_channels,
                full_out_time,
                gamma,
                beta,
            );
        } else if attrs.norm.has_layer_norm(index) {
            let gamma = layer_weights.norm_gamma.as_ref().expect("validated gamma");
            let beta = layer_weights.norm_beta.as_ref().expect("validated beta");
            let mut frame_major =
                transpose_channel_to_frame(&convolution, layer.out_channels, full_out_time);
            layer_norm_with_compute_inplace(
                &mut frame_major,
                full_out_time,
                layer.out_channels,
                gamma,
                beta,
                1e-5,
                compute,
            )?;
            convolution =
                transpose_frame_to_channel(&frame_major, full_out_time, layer.out_channels);
        }

        let mut activated = vec![0.0f32; convolution.len()];
        compute.gelu_f32(&convolution, &mut activated)?;
        current = vec![0.0f32; layer.out_channels * valid_out_time];
        for channel in 0..layer.out_channels {
            current[channel * valid_out_time..(channel + 1) * valid_out_time].copy_from_slice(
                &activated[channel * full_out_time..channel * full_out_time + valid_out_time],
            );
        }
        valid_time = valid_out_time;
        in_channels = layer.out_channels;
    }
    Ok(transpose_channel_to_frame(
        &current,
        in_channels,
        valid_time,
    ))
}

fn group_norm_each_channel(
    values: &mut [f32],
    channels: usize,
    time: usize,
    gamma: &[f32],
    beta: &[f32],
) {
    for channel in 0..channels {
        let row = &mut values[channel * time..(channel + 1) * time];
        let mean = row.iter().copied().sum::<f32>() / time as f32;
        let variance = row
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f32>()
            / time as f32;
        let inverse = 1.0 / (variance + 1e-5).sqrt();
        for value in row {
            *value = (*value - mean) * inverse * gamma[channel] + beta[channel];
        }
    }
}

fn transpose_channel_to_frame(values: &[f32], channels: usize, time: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; values.len()];
    for channel in 0..channels {
        for frame in 0..time {
            output[frame * channels + channel] = values[channel * time + frame];
        }
    }
    output
}

fn transpose_frame_to_channel(values: &[f32], time: usize, channels: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; values.len()];
    for frame in 0..time {
        for channel in 0..channels {
            output[channel * time + frame] = values[frame * channels + channel];
        }
    }
    output
}

fn endpoint_head(hidden: &[f32], frames: usize, head: &EndpointHead) -> f32 {
    let mut scores = Vec::with_capacity(frames);
    for frame in hidden.chunks_exact(HIDDEN) {
        let mut projected = linear_forward(frame, 1, HIDDEN, &head.pool_w1, &head.pool_b1, 256);
        for value in &mut projected {
            *value = value.tanh();
        }
        scores.push(linear_forward(&projected, 1, 256, &head.pool_w2, &head.pool_b2, 1)[0]);
    }
    stable_softmax(&mut scores);
    let mut pooled = vec![0.0f32; HIDDEN];
    for (frame, &weight) in hidden.chunks_exact(HIDDEN).zip(&scores) {
        for (dst, &value) in pooled.iter_mut().zip(frame) {
            *dst += value * weight;
        }
    }

    let mut value = linear_forward(
        &pooled,
        1,
        HIDDEN,
        &head.classifier_w1,
        &head.classifier_b1,
        256,
    );
    layer_norm_inplace(
        &mut value,
        1,
        256,
        &head.classifier_norm_gamma,
        &head.classifier_norm_beta,
        1e-5,
    );
    for element in &mut value {
        *element = gelu_exact(*element);
    }
    let mut value = linear_forward(&value, 1, 256, &head.classifier_w2, &head.classifier_b2, 64);
    for element in &mut value {
        *element = gelu_exact(*element);
    }
    let logit = linear_forward(&value, 1, 64, &head.classifier_w3, &head.classifier_b3, 1)[0];
    sigmoid(logit)
}

fn endpoint_head_with_compute(
    hidden: &[f32],
    frames: usize,
    head: &EndpointHead,
    compute: &Compute,
) -> Result<f32> {
    let mut projected = linear_forward_with_compute(
        hidden,
        frames,
        HIDDEN,
        &head.pool_w1,
        &head.pool_b1,
        256,
        compute,
    )?;
    for value in &mut projected {
        *value = value.tanh();
    }
    let scores = linear_forward_with_compute(
        &projected,
        frames,
        256,
        &head.pool_w2,
        &head.pool_b2,
        1,
        compute,
    )?;
    let mut attention = vec![0.0f32; frames];
    compute.softmax_f32(&scores, &mut attention, 1, frames)?;
    let mut pooled = vec![0.0f32; HIDDEN];
    compute.gemm_f32(1, HIDDEN, frames, &attention, hidden, None, &mut pooled)?;

    let mut value = linear_forward_with_compute(
        &pooled,
        1,
        HIDDEN,
        &head.classifier_w1,
        &head.classifier_b1,
        256,
        compute,
    )?;
    layer_norm_with_compute_inplace(
        &mut value,
        1,
        256,
        &head.classifier_norm_gamma,
        &head.classifier_norm_beta,
        1e-5,
        compute,
    )?;
    let mut activated = vec![0.0f32; value.len()];
    compute.gelu_f32(&value, &mut activated)?;
    value = linear_forward_with_compute(
        &activated,
        1,
        256,
        &head.classifier_w2,
        &head.classifier_b2,
        64,
        compute,
    )?;
    activated.resize(value.len(), 0.0);
    compute.gelu_f32(&value, &mut activated)?;
    let logit = linear_forward_with_compute(
        &activated,
        1,
        64,
        &head.classifier_w3,
        &head.classifier_b3,
        1,
        compute,
    )?[0];
    Ok(sigmoid(logit))
}

fn zero_mean_unit_var(pcm: &[f32], eps: f32) -> Vec<f32> {
    let mean = pcm.iter().copied().sum::<f32>() / pcm.len() as f32;
    let variance = pcm
        .iter()
        .map(|&value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f32>()
        / pcm.len() as f32;
    let scale = 1.0 / (variance + eps).sqrt();
    pcm.iter().map(|&value| (value - mean) * scale).collect()
}

fn ratio_mask_frames(input_len: usize, padded_len: usize, full_frames: usize) -> usize {
    // PyTorch promotes `torch.arange(i64) * Python-float` to the default
    // floating dtype (F32 here) before `.long()` truncation. Preserve that
    // rounding at the handful of exact input-index boundaries.
    let ratio = padded_len as f32 / full_frames as f32;
    (0..full_frames)
        .take_while(|&index| ((index as f32) * ratio).trunc() < input_len as f32)
        .count()
}

fn stable_softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn tensor(file: &GgufFile, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("smart_turn GGUF missing tensor `{name}`")))?;
    let expected: Vec<u64> = dims.iter().map(|&dim| dim as u64).collect();
    if info.dimensions != expected || info.dtype != GgmlType::F32 {
        return Err(VokraError::ModelLoad(format!(
            "smart_turn GGUF tensor `{name}` is {:?} {:?}, expected F32 {expected:?}",
            info.dtype, info.dimensions
        )));
    }
    file.tensor_f32(name)
        .map_err(|error| VokraError::ModelLoad(format!("smart_turn reading `{name}`: {error}")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("smart_turn GGUF missing string `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "smart_turn GGUF `{key}` is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn optional_string(file: &GgufFile, key: &str) -> Option<String> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .map(str::to_owned)
}

fn meta_u32(file: &GgufFile, key: &str) -> Result<u32> {
    let raw = file
        .get(key)
        .and_then(GgufMetadataValue::as_u64)
        .ok_or_else(|| VokraError::ModelLoad(format!("smart_turn GGUF missing u32 `{key}`")))?;
    u32::try_from(raw)
        .map_err(|_| VokraError::ModelLoad(format!("smart_turn GGUF `{key}` overflows u32")))
}

fn meta_f32(file: &GgufFile, key: &str) -> Result<f32> {
    file.get(key)
        .and_then(GgufMetadataValue::as_f64)
        .map(|value| value as f32)
        .ok_or_else(|| VokraError::ModelLoad(format!("smart_turn GGUF missing float `{key}`")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipecat_ratio_mask_keeps_only_the_documented_small_suffix() {
        let attrs = WaveformFrontendAttrs::wav2vec2_base();
        let full = attrs.predict_t_out(MAX_INPUT_SAMPLES).unwrap();
        let valid = attrs.predict_t_out(16_000).unwrap();
        assert_eq!(full, 799);
        assert_eq!(valid, 49);
        assert_eq!(ratio_mask_frames(16_000, MAX_INPUT_SAMPLES, full), 50);
        assert_eq!(ratio_mask_frames(123_995, MAX_INPUT_SAMPLES, full), 387);

        let mut max_extra_queries = 0;
        for samples in 400..=MAX_INPUT_SAMPLES {
            let valid = attrs.predict_t_out(samples).unwrap();
            let pooled = ratio_mask_frames(samples, MAX_INPUT_SAMPLES, full);
            assert!(pooled >= valid);
            max_extra_queries = max_extra_queries.max(pooled - valid);
        }
        assert_eq!(max_extra_queries, 2);
    }

    #[test]
    fn real_checkpoint_compute_cpu_matches_scalar_when_supplied() {
        let Some(path) = std::env::var_os("VOKRA_SMART_TURN_GGUF") else {
            eprintln!("skip SmartTurn complete Compute parity: VOKRA_SMART_TURN_GGUF unset");
            return;
        };
        let model = SmartTurn::from_path(path).expect("bind canonical SmartTurn GGUF");
        let pcm: Vec<f32> = (0..16_000)
            .map(|sample| {
                0.25 * (2.0 * std::f32::consts::PI * 220.0 * sample as f32 / 16_000.0).sin()
            })
            .collect();
        let scalar = model
            .predict_endpoint(&pcm, 16_000)
            .expect("scalar SmartTurn")
            .completion_probability();
        let normalized = zero_mean_unit_var(&pcm, NORMALIZATION_EPS);
        let compute = Compute::cpu();
        let dispatched = model
            .predict_endpoint_validated(pcm.len(), &normalized, Some(&compute))
            .expect("complete Compute SmartTurn")
            .completion_probability();
        let max_abs = (scalar - dispatched).abs();
        assert!(
            max_abs <= 1.0e-4,
            "SmartTurn scalar/Compute probability max_abs={max_abs:.9e} exceeds 1e-4"
        );
    }

    #[test]
    fn normalization_is_zero_mean_unit_variance() {
        let normalized = zero_mean_unit_var(&[-2.0, -1.0, 1.0, 2.0], NORMALIZATION_EPS);
        let mean = normalized.iter().sum::<f32>() / normalized.len() as f32;
        let variance =
            normalized.iter().map(|value| value * value).sum::<f32>() / normalized.len() as f32;
        assert!(mean.abs() < 1e-7);
        assert!((variance - 1.0).abs() < 1e-5);
    }

    #[test]
    fn turn_prediction_validates_probability() {
        let prediction = TurnPrediction::new(0.75).unwrap();
        assert!(prediction.is_complete(DEFAULT_COMPLETION_THRESHOLD));
        assert!(TurnPrediction::new(f32::NAN).is_err());
        assert!(TurnPrediction::new(1.01).is_err());
    }

    #[test]
    fn softmax_and_sigmoid_are_stable() {
        let mut values = [1000.0, 999.0, -1000.0];
        stable_softmax(&mut values);
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(sigmoid(1000.0), 1.0);
        assert_eq!(sigmoid(-1000.0), 0.0);
    }
}
