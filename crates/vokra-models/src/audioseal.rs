//! Native AudioSeal 0.2 generator/detector runtime for CPU and Metal.
//!
//! The public Vokra artifact is a single 310-tensor bundle containing the
//! official base and streaming generator/detector checkpoints.  This module
//! binds that complete manifest, reconstructs PyTorch weight normalization,
//! and executes the exact SEANet + two-layer LSTM topology.  Every learned
//! convolution, transposed convolution projection, recurrent projection and
//! detector softmax is dispatched through [`Compute`].  ELU, padding,
//! residual addition, message lookup and scalar thresholding are host-side
//! tensor-layout glue, not hidden CPU model operators.
//!
//! Selecting a backend first validates the complete [`AUDIOSEAL_HOT_OPS`]
//! registry.  Metal therefore runs the full learned graph through real Metal
//! kernels; an uncovered backend returns an explicit error before inference
//! and never falls back to CPU.
//!
//! [`AudiosealVariant::Streaming`] selects the causal streaming-trained
//! weights.  The methods in this initial binder evaluate a complete buffer,
//! matching upstream's non-context-manager `forward`; a state-carrying chunk
//! API is intentionally not claimed by this type.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, GgufValueType, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};

/// Converter/runtime architecture handshake.
pub const ARCH: &str = "audioseal_real_weight";
/// Canonical public model name.
pub const NAME: &str = "audioseal_real_weight";
/// Official upstream model repository.
pub const UPSTREAM_HF: &str = "facebook/audioseal";
/// Immutable revision containing all four checkpoint files.
pub const CHECKPOINT_REVISION: &str = "3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba";
/// Immutable source revision used to transcribe the forward.
pub const SOURCE_REVISION: &str = "e63a8a0e5cdf7bb797159c92ba15961557fe9bd2";
/// Required PCM sample rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// Number of embedded/recovered message bits.
pub const NBITS: usize = 16;
/// Exact tensor count for four checkpoint variants.
pub const TENSOR_COUNT: usize = 310;

const CATEGORY: &str = "watermark";
const DIMENSION: usize = 128;
const N_FILTERS: usize = 32;
const LSTM_DIM: usize = 512;
const DETECTOR_DIM: usize = 32;
const DETECTOR_CHANNELS: usize = 2 + NBITS;
const HOP_LENGTH: usize = 320;
const RATIOS: [usize; 4] = [8, 5, 4, 2];

const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const PREFIX: &str = "vokra.audioseal";
const KEY_CHECKPOINT_REVISION: &str = "vokra.audioseal.checkpoint_revision";
const KEY_SOURCE_REVISION: &str = "vokra.audioseal.source_revision";
const KEY_SAMPLE_RATE: &str = "vokra.audioseal.sample_rate";
const KEY_NBITS: &str = "vokra.audioseal.nbits";
const KEY_CHANNELS: &str = "vokra.audioseal.channels";
const KEY_DIMENSION: &str = "vokra.audioseal.dimension";
const KEY_N_FILTERS: &str = "vokra.audioseal.n_filters";
const KEY_N_RESIDUAL_LAYERS: &str = "vokra.audioseal.n_residual_layers";
const KEY_RATIOS: &str = "vokra.audioseal.ratios";
const KEY_ACTIVATION: &str = "vokra.audioseal.activation";
const KEY_COMPRESS: &str = "vokra.audioseal.compress";
const KEY_DILATION_BASE: &str = "vokra.audioseal.dilation_base";
const KEY_KERNEL_SIZE: &str = "vokra.audioseal.kernel_size";
const KEY_LAST_KERNEL_SIZE: &str = "vokra.audioseal.last_kernel_size";
const KEY_RESIDUAL_KERNEL_SIZE: &str = "vokra.audioseal.residual_kernel_size";
const KEY_LSTM_LAYERS: &str = "vokra.audioseal.lstm_layers";
const KEY_NORM: &str = "vokra.audioseal.norm";
const KEY_PAD_MODE: &str = "vokra.audioseal.pad_mode";
const KEY_TRUE_SKIP: &str = "vokra.audioseal.true_skip";
const KEY_BASE_CAUSAL: &str = "vokra.audioseal.base_causal";
const KEY_STREAMING_CAUSAL: &str = "vokra.audioseal.streaming_causal";
const KEY_DETECTOR_OUTPUT_DIM: &str = "vokra.audioseal.detector_output_dim";
const KEY_HOP_LENGTH: &str = "vokra.audioseal.hop_length";
const KEY_NORMALIZER: &str = "vokra.audioseal.normalizer";

const CONTRACT_KEYS: &[&str] = &[
    KEY_CHECKPOINT_REVISION,
    KEY_SOURCE_REVISION,
    KEY_SAMPLE_RATE,
    KEY_NBITS,
    KEY_CHANNELS,
    KEY_DIMENSION,
    KEY_N_FILTERS,
    KEY_N_RESIDUAL_LAYERS,
    KEY_RATIOS,
    KEY_ACTIVATION,
    KEY_COMPRESS,
    KEY_DILATION_BASE,
    KEY_KERNEL_SIZE,
    KEY_LAST_KERNEL_SIZE,
    KEY_RESIDUAL_KERNEL_SIZE,
    KEY_LSTM_LAYERS,
    KEY_NORM,
    KEY_PAD_MODE,
    KEY_TRUE_SKIP,
    KEY_BASE_CAUSAL,
    KEY_STREAMING_CAUSAL,
    KEY_DETECTOR_OUTPUT_DIM,
    KEY_HOP_LENGTH,
    KEY_NORMALIZER,
];

const LEGACY_SOURCE: &str = "facebook/audioseal (paired Generator + Detector 16-bit-message audio watermark for EU AI Act Article 50 compliance, San Roman et al. 2024 ICML arXiv:2401.17264, MIT — runtime binder gated on M5-05 T04 ADR ratification)";

/// Complete learned-op registry for both generator and detector variants.
pub const AUDIOSEAL_HOT_OPS: &[HotOp] = &[HotOp::Conv1d, HotOp::Gemm, HotOp::Gemv, HotOp::Softmax];

/// Selects one of the two official topology-compatible checkpoint variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudiosealVariant {
    /// Non-causal base checkpoint.
    #[default]
    Base,
    /// Causal streaming-trained checkpoint, evaluated as one complete buffer.
    Streaming,
}

/// AudioSeal detector output.
#[derive(Debug, Clone, PartialEq)]
pub struct AudiosealDetection {
    /// Fraction of samples whose watermarked-class probability exceeds the
    /// requested detection threshold, matching upstream `detect_watermark`.
    pub detection_probability: f32,
    /// Per-sample watermarked-class probability.
    pub positive_probabilities: Vec<f32>,
    /// Mean-logit sigmoid probabilities for the 16 message bits.
    pub message_probabilities: [f32; NBITS],
    /// Thresholded message bits.
    pub message: [u8; NBITS],
}

#[derive(Debug)]
struct BundleWeights {
    generator_base: Generator,
    generator_streaming: Generator,
    detector_base: Detector,
    detector_streaming: Detector,
}

/// Strict native AudioSeal bundle.
#[derive(Debug, Clone)]
pub struct Audioseal {
    weights: Arc<BundleWeights>,
    backend: BackendKind,
    legacy_metadata_repaired: bool,
}

impl Audioseal {
    /// Opens and binds an AudioSeal GGUF.
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let file = GgufFile::open(path)?;
        Self::from_file(&file)
    }

    /// Strictly binds an already-open GGUF.
    pub fn from_file(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        require_string(file, chunks::KEY_MODEL_NAME, NAME)?;
        require_string(file, KEY_CATEGORY, CATEGORY)?;
        require_string(file, KEY_UPSTREAM_HF, UPSTREAM_HF)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "mit")?;
        let license = vokra_core::resolve_license_class(file);
        if license.class != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: weight license resolves to {}, expected Permissive for official MIT weights",
                license.class.as_str()
            )));
        }

        validate_manifest(file)?;
        let present = CONTRACT_KEYS
            .iter()
            .filter(|key| file.get(key).is_some())
            .count();
        let legacy_metadata_repaired = match present {
            0 => {
                require_string(file, chunks::KEY_PROVENANCE_SOURCE, LEGACY_SOURCE)?;
                true
            }
            count if count == CONTRACT_KEYS.len() => {
                validate_contract_metadata(file)?;
                false
            }
            count => {
                return Err(VokraError::ModelLoad(format!(
                    "{ARCH}: partial `{PREFIX}.*` metadata ({count}/{} keys); refusing topology repair",
                    CONTRACT_KEYS.len()
                )));
            }
        };

        let weights = BundleWeights {
            generator_base: Generator::load(file, "generator_base", false)?,
            generator_streaming: Generator::load(file, "generator_streaming", true)?,
            detector_base: Detector::load(file, "detector_base", false, WeightNames::Legacy)?,
            detector_streaming: Detector::load(
                file,
                "detector_streaming",
                true,
                WeightNames::Parametrized,
            )?,
        };
        Ok(Self {
            weights: Arc::new(weights),
            backend: BackendKind::Cpu,
            legacy_metadata_repaired,
        })
    }

    /// Selects the backend for the entire learned graph.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Selected backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Whether the exact historical public header required fixed-topology
    /// metadata repair after its complete 310-tensor manifest matched.
    #[must_use]
    pub const fn legacy_metadata_repaired(&self) -> bool {
        self.legacy_metadata_repaired
    }

    /// Returns the raw watermark waveform before it is mixed into `pcm`.
    pub fn watermark_pcm(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        message: &[u8; NBITS],
        variant: AudiosealVariant,
    ) -> Result<Vec<f32>> {
        validate_pcm(pcm, sample_rate)?;
        validate_message(message)?;
        let compute = Compute::for_backend(self.backend, AUDIOSEAL_HOT_OPS)?;
        self.generator(variant).watermark(pcm, message, &compute)
    }

    /// Embeds a 16-bit message and returns `pcm + alpha * watermark`.
    pub fn embed_pcm(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        message: &[u8; NBITS],
        alpha: f32,
        variant: AudiosealVariant,
    ) -> Result<Vec<f32>> {
        if !alpha.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: alpha must be finite, got {alpha}"
            )));
        }
        let watermark = self.watermark_pcm(pcm, sample_rate, message, variant)?;
        let output = pcm
            .iter()
            .zip(watermark)
            .map(|(&sample, watermark)| sample + alpha * watermark)
            .collect::<Vec<_>>();
        reject_non_finite("embedded waveform", &output)?;
        Ok(output)
    }

    /// Detects a watermark with upstream's default 0.5 thresholds.
    pub fn detect_pcm(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        variant: AudiosealVariant,
    ) -> Result<AudiosealDetection> {
        self.detect_pcm_with_thresholds(pcm, sample_rate, variant, 0.5, 0.5)
    }

    /// Detects a watermark with explicit per-sample and message thresholds.
    pub fn detect_pcm_with_thresholds(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        variant: AudiosealVariant,
        detection_threshold: f32,
        message_threshold: f32,
    ) -> Result<AudiosealDetection> {
        validate_pcm(pcm, sample_rate)?;
        validate_threshold("detection_threshold", detection_threshold)?;
        validate_threshold("message_threshold", message_threshold)?;
        let compute = Compute::for_backend(self.backend, AUDIOSEAL_HOT_OPS)?;
        self.detector(variant)
            .detect(pcm, detection_threshold, message_threshold, &compute)
    }

    fn generator(&self, variant: AudiosealVariant) -> &Generator {
        match variant {
            AudiosealVariant::Base => &self.weights.generator_base,
            AudiosealVariant::Streaming => &self.weights.generator_streaming,
        }
    }

    fn detector(&self, variant: AudiosealVariant) -> &Detector {
        match variant {
            AudiosealVariant::Base => &self.weights.detector_base,
            AudiosealVariant::Streaming => &self.weights.detector_streaming,
        }
    }
}

#[derive(Debug)]
struct Generator {
    encoder: SeanetEncoder,
    decoder: SeanetDecoder,
    message_embedding: Vec<f32>,
}

impl Generator {
    fn load(file: &GgufFile, root: &str, causal: bool) -> Result<Self> {
        Ok(Self {
            encoder: SeanetEncoder::load(
                file,
                &format!("{root}.encoder"),
                causal,
                WeightNames::Legacy,
            )?,
            decoder: SeanetDecoder::load(
                file,
                &format!("{root}.decoder"),
                causal,
                WeightNames::Legacy,
            )?,
            message_embedding: tensor(
                file,
                &format!("{root}.msg_processor.msg_processor.weight"),
                &[2 * NBITS, DIMENSION],
            )?,
        })
    }

    fn watermark(&self, pcm: &[f32], message: &[u8; NBITS], compute: &Compute) -> Result<Vec<f32>> {
        let (mut hidden, frames) = self.encoder.forward(pcm, pcm.len(), compute)?;
        let mut message_vector = [0.0f32; DIMENSION];
        for (bit_index, &bit) in message.iter().enumerate() {
            let row = 2 * bit_index + usize::from(bit);
            for (dim, value) in message_vector.iter_mut().enumerate() {
                *value += self.message_embedding[row * DIMENSION + dim];
            }
        }
        for channel in 0..DIMENSION {
            let addition = message_vector[channel];
            for value in &mut hidden[channel * frames..(channel + 1) * frames] {
                *value += addition;
            }
        }
        let (mut waveform, decoded_len) = self.decoder.forward(&hidden, frames, compute)?;
        if decoded_len < pcm.len() {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: decoder produced {decoded_len} samples for {}-sample input",
                pcm.len()
            )));
        }
        waveform.truncate(pcm.len());
        reject_non_finite("watermark waveform", &waveform)?;
        Ok(waveform)
    }
}

#[derive(Debug)]
struct Detector {
    encoder: SeanetEncoder,
    reverse: ConvTranspose1d,
    head: Conv1d,
}

impl Detector {
    fn load(file: &GgufFile, root: &str, causal: bool, names: WeightNames) -> Result<Self> {
        let encoder_prefix = format!("{root}.detector.0");
        Ok(Self {
            encoder: SeanetEncoder::load(file, &encoder_prefix, causal, names)?,
            reverse: ConvTranspose1d::load_raw(
                file,
                &format!("{encoder_prefix}.reverse_convolution"),
                DIMENSION,
                DETECTOR_DIM,
                HOP_LENGTH,
                HOP_LENGTH,
            )?,
            head: Conv1d::load_raw(
                file,
                &format!("{root}.detector.1"),
                DETECTOR_DIM,
                DETECTOR_CHANNELS,
                1,
                1,
            )?,
        })
    }

    fn detect(
        &self,
        pcm: &[f32],
        detection_threshold: f32,
        message_threshold: f32,
        compute: &Compute,
    ) -> Result<AudiosealDetection> {
        let (encoded, frames) = self.encoder.forward(pcm, pcm.len(), compute)?;
        let (mut restored, restored_len) = self.reverse.forward_raw(&encoded, frames, compute)?;
        if restored_len < pcm.len() {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: detector reverse convolution produced {restored_len} samples for {}-sample input",
                pcm.len()
            )));
        }
        if restored_len != pcm.len() {
            let mut trimmed = vec![0.0f32; DETECTOR_DIM * pcm.len()];
            for channel in 0..DETECTOR_DIM {
                trimmed[channel * pcm.len()..(channel + 1) * pcm.len()].copy_from_slice(
                    &restored[channel * restored_len..channel * restored_len + pcm.len()],
                );
            }
            restored = trimmed;
        }
        let (logits, samples) = self.head.forward(&restored, pcm.len(), false, compute)?;
        debug_assert_eq!(samples, pcm.len());

        let mut binary_logits = vec![0.0f32; samples * 2];
        for sample in 0..samples {
            binary_logits[2 * sample] = logits[sample];
            binary_logits[2 * sample + 1] = logits[samples + sample];
        }
        let mut binary_probabilities = vec![0.0f32; binary_logits.len()];
        compute.softmax_f32(&binary_logits, &mut binary_probabilities, samples, 2)?;
        let positive_probabilities = (0..samples)
            .map(|sample| binary_probabilities[2 * sample + 1])
            .collect::<Vec<_>>();
        let detection_probability = positive_probabilities
            .iter()
            .filter(|&&probability| probability > detection_threshold)
            .count() as f32
            / samples as f32;

        let mut message_probabilities = [0.0f32; NBITS];
        let mut message = [0u8; NBITS];
        for bit in 0..NBITS {
            let channel = bit + 2;
            let sum = logits[channel * samples..(channel + 1) * samples]
                .iter()
                .sum::<f32>();
            let probability = sigmoid(sum / samples as f32);
            message_probabilities[bit] = probability;
            message[bit] = u8::from(probability > message_threshold);
        }
        reject_non_finite("detector positive probabilities", &positive_probabilities)?;
        reject_non_finite("detector message probabilities", &message_probabilities)?;
        Ok(AudiosealDetection {
            detection_probability,
            positive_probabilities,
            message_probabilities,
            message,
        })
    }
}

#[derive(Debug)]
struct SeanetEncoder {
    causal: bool,
    initial: Conv1d,
    residuals: Vec<ResidualBlock>,
    downsample: Vec<Conv1d>,
    lstm: Lstm,
    final_conv: Conv1d,
}

impl SeanetEncoder {
    fn load(file: &GgufFile, prefix: &str, causal: bool, names: WeightNames) -> Result<Self> {
        let initial = Conv1d::load_weight_norm(
            file,
            &format!("{prefix}.model.0.conv.conv"),
            1,
            32,
            7,
            1,
            names,
        )?;
        let mut residuals = Vec::with_capacity(4);
        let mut downsample = Vec::with_capacity(4);
        for (residual, down, channels, next, kernel, stride) in [
            (1, 3, 32, 64, 4, 2),
            (4, 6, 64, 128, 8, 4),
            (7, 9, 128, 256, 10, 5),
            (10, 12, 256, 512, 16, 8),
        ] {
            residuals.push(ResidualBlock::load(
                file, prefix, residual, channels, names,
            )?);
            downsample.push(Conv1d::load_weight_norm(
                file,
                &format!("{prefix}.model.{down}.conv.conv"),
                channels,
                next,
                kernel,
                stride,
                names,
            )?);
        }
        Ok(Self {
            causal,
            initial,
            residuals,
            downsample,
            lstm: Lstm::load(file, &format!("{prefix}.model.13.lstm"), LSTM_DIM, 2)?,
            final_conv: Conv1d::load_weight_norm(
                file,
                &format!("{prefix}.model.15.conv.conv"),
                512,
                DIMENSION,
                7,
                1,
                names,
            )?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        input_len: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        let (mut hidden, mut time) =
            self.initial
                .forward(input, input_len, self.causal, compute)?;
        for stage in 0..4 {
            hidden = self.residuals[stage].forward(&hidden, time, self.causal, compute)?;
            elu_inplace(&mut hidden);
            (hidden, time) = self.downsample[stage].forward(&hidden, time, self.causal, compute)?;
        }
        hidden = self.lstm.forward(&hidden, time, compute)?;
        elu_inplace(&mut hidden);
        self.final_conv.forward(&hidden, time, self.causal, compute)
    }
}

#[derive(Debug)]
struct SeanetDecoder {
    causal: bool,
    initial: Conv1d,
    lstm: Lstm,
    upsample: Vec<ConvTranspose1d>,
    residuals: Vec<ResidualBlock>,
    final_conv: Conv1d,
}

impl SeanetDecoder {
    fn load(file: &GgufFile, prefix: &str, causal: bool, names: WeightNames) -> Result<Self> {
        let initial = Conv1d::load_weight_norm(
            file,
            &format!("{prefix}.model.0.conv.conv"),
            DIMENSION,
            512,
            7,
            1,
            names,
        )?;
        let mut upsample = Vec::with_capacity(4);
        let mut residuals = Vec::with_capacity(4);
        for (transpose, residual, channels, next, kernel, stride) in [
            (3, 4, 512, 256, 16, 8),
            (6, 7, 256, 128, 10, 5),
            (9, 10, 128, 64, 8, 4),
            (12, 13, 64, 32, 4, 2),
        ] {
            upsample.push(ConvTranspose1d::load_weight_norm(
                file,
                &format!("{prefix}.model.{transpose}.convtr.convtr"),
                channels,
                next,
                kernel,
                stride,
                names,
            )?);
            residuals.push(ResidualBlock::load(file, prefix, residual, next, names)?);
        }
        Ok(Self {
            causal,
            initial,
            lstm: Lstm::load(file, &format!("{prefix}.model.1.lstm"), LSTM_DIM, 2)?,
            upsample,
            residuals,
            final_conv: Conv1d::load_weight_norm(
                file,
                &format!("{prefix}.model.15.conv.conv"),
                32,
                1,
                7,
                1,
                names,
            )?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        input_len: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        let (mut hidden, mut time) =
            self.initial
                .forward(input, input_len, self.causal, compute)?;
        hidden = self.lstm.forward(&hidden, time, compute)?;
        for stage in 0..4 {
            elu_inplace(&mut hidden);
            (hidden, time) = self.upsample[stage].forward(&hidden, time, self.causal, compute)?;
            hidden = self.residuals[stage].forward(&hidden, time, self.causal, compute)?;
        }
        elu_inplace(&mut hidden);
        self.final_conv.forward(&hidden, time, self.causal, compute)
    }
}

#[derive(Debug)]
struct ResidualBlock {
    first: Conv1d,
    second: Conv1d,
}

impl ResidualBlock {
    fn load(
        file: &GgufFile,
        prefix: &str,
        index: usize,
        channels: usize,
        names: WeightNames,
    ) -> Result<Self> {
        let hidden = channels / 2;
        Ok(Self {
            first: Conv1d::load_weight_norm(
                file,
                &format!("{prefix}.model.{index}.block.1.conv.conv"),
                channels,
                hidden,
                3,
                1,
                names,
            )?,
            second: Conv1d::load_weight_norm(
                file,
                &format!("{prefix}.model.{index}.block.3.conv.conv"),
                hidden,
                channels,
                1,
                1,
                names,
            )?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        time: usize,
        causal: bool,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        let residual = input;
        let mut hidden = input.to_vec();
        elu_inplace(&mut hidden);
        let (mut hidden, first_time) = self.first.forward(&hidden, time, causal, compute)?;
        if first_time != time {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: residual k=3 convolution changed time {time} -> {first_time}"
            )));
        }
        elu_inplace(&mut hidden);
        let (mut hidden, second_time) = self.second.forward(&hidden, time, causal, compute)?;
        if second_time != time || hidden.len() != residual.len() {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: residual branch shape mismatch (time {second_time}, values {}) vs input (time {time}, values {})",
                hidden.len(),
                residual.len()
            )));
        }
        for (value, &skip) in hidden.iter_mut().zip(residual) {
            *value += skip;
        }
        Ok(hidden)
    }
}

#[derive(Debug)]
struct Conv1d {
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Conv1d {
    #[allow(clippy::too_many_arguments)]
    fn load_weight_norm(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
        names: WeightNames,
    ) -> Result<Self> {
        let (g_name, v_name) = names.names(prefix);
        let g = tensor(file, &g_name, &[output_channels, 1, 1])?;
        let v = tensor(file, &v_name, &[output_channels, input_channels, kernel])?;
        let weight =
            reconstruct_weight_norm(&g, &v, output_channels, input_channels * kernel, prefix)?;
        Ok(Self {
            input_channels,
            output_channels,
            kernel,
            stride,
            weight,
            bias: tensor(file, &format!("{prefix}.bias"), &[output_channels])?,
        })
    }

    fn load_raw(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self> {
        Ok(Self {
            input_channels,
            output_channels,
            kernel,
            stride,
            weight: tensor(
                file,
                &format!("{prefix}.weight"),
                &[output_channels, input_channels, kernel],
            )?,
            bias: tensor(file, &format!("{prefix}.bias"), &[output_channels])?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        input_len: usize,
        causal: bool,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        require_layout("conv1d input", input, self.input_channels, input_len)?;
        if self.kernel < self.stride || self.stride == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: invalid conv kernel={} stride={}",
                self.kernel, self.stride
            )));
        }
        let padding_total = self.kernel - self.stride;
        let extra = (self.stride - input_len % self.stride) % self.stride;
        let padding_right = if causal {
            extra
        } else {
            padding_total / 2 + extra
        };
        let padding_left = if causal {
            padding_total
        } else {
            padding_total - padding_total / 2
        };
        let padded_len = padding_left + input_len + padding_right;
        let mut padded = vec![0.0f32; self.input_channels * padded_len];
        for channel in 0..self.input_channels {
            padded[channel * padded_len + padding_left
                ..channel * padded_len + padding_left + input_len]
                .copy_from_slice(&input[channel * input_len..(channel + 1) * input_len]);
        }
        let output_len = (padded_len - self.kernel) / self.stride + 1;
        let mut output = vec![0.0f32; self.output_channels * output_len];
        compute.conv1d_f32(
            &padded,
            self.input_channels,
            padded_len,
            &self.weight,
            self.output_channels,
            self.kernel,
            Some(&self.bias),
            self.stride,
            0,
            &mut output,
        )?;
        Ok((output, output_len))
    }
}

#[derive(Debug)]
struct ConvTranspose1d {
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    /// PyTorch layout `[input_channels, output_channels, kernel]`, which is
    /// also a row-major GEMM matrix `[input_channels, output_channels*kernel]`.
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl ConvTranspose1d {
    #[allow(clippy::too_many_arguments)]
    fn load_weight_norm(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
        names: WeightNames,
    ) -> Result<Self> {
        let (g_name, v_name) = names.names(prefix);
        let g = tensor(file, &g_name, &[input_channels, 1, 1])?;
        let v = tensor(file, &v_name, &[input_channels, output_channels, kernel])?;
        let weight =
            reconstruct_weight_norm(&g, &v, input_channels, output_channels * kernel, prefix)?;
        Ok(Self {
            input_channels,
            output_channels,
            kernel,
            stride,
            weight,
            bias: tensor(file, &format!("{prefix}.bias"), &[output_channels])?,
        })
    }

    fn load_raw(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self> {
        Ok(Self {
            input_channels,
            output_channels,
            kernel,
            stride,
            weight: tensor(
                file,
                &format!("{prefix}.weight"),
                &[input_channels, output_channels, kernel],
            )?,
            bias: tensor(file, &format!("{prefix}.bias"), &[output_channels])?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        input_len: usize,
        causal: bool,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        let (raw, raw_len) = self.forward_raw(input, input_len, compute)?;
        let padding_total = self.kernel.checked_sub(self.stride).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "{ARCH}: conv-transpose kernel {} < stride {}",
                self.kernel, self.stride
            ))
        })?;
        let padding_right = if causal {
            padding_total
        } else {
            padding_total / 2
        };
        let padding_left = if causal {
            0
        } else {
            padding_total - padding_right
        };
        let output_len = raw_len
            .checked_sub(padding_left + padding_right)
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "{ARCH}: conv-transpose raw length {raw_len} smaller than trim {}",
                    padding_left + padding_right
                ))
            })?;
        let mut output = vec![0.0f32; self.output_channels * output_len];
        for channel in 0..self.output_channels {
            output[channel * output_len..(channel + 1) * output_len].copy_from_slice(
                &raw[channel * raw_len + padding_left
                    ..channel * raw_len + padding_left + output_len],
            );
        }
        Ok((output, output_len))
    }

    fn forward_raw(
        &self,
        input: &[f32],
        input_len: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        require_layout(
            "conv-transpose input",
            input,
            self.input_channels,
            input_len,
        )?;
        if input_len == 0 || self.stride == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: conv-transpose requires non-empty input and stride > 0"
            )));
        }
        let raw_len = (input_len - 1)
            .checked_mul(self.stride)
            .and_then(|value| value.checked_add(self.kernel))
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "{ARCH}: conv-transpose output length overflow"
                ))
            })?;
        let mut time_major = vec![0.0f32; input_len * self.input_channels];
        for time in 0..input_len {
            for channel in 0..self.input_channels {
                time_major[time * self.input_channels + channel] =
                    input[channel * input_len + time];
            }
        }
        let projected_width = self.output_channels * self.kernel;
        let mut projected = vec![0.0f32; input_len * projected_width];
        compute.gemm_f32(
            input_len,
            projected_width,
            self.input_channels,
            &time_major,
            &self.weight,
            None,
            &mut projected,
        )?;
        let mut output = vec![0.0f32; self.output_channels * raw_len];
        for channel in 0..self.output_channels {
            output[channel * raw_len..(channel + 1) * raw_len].fill(self.bias[channel]);
        }
        for time in 0..input_len {
            let destination = time * self.stride;
            for channel in 0..self.output_channels {
                let source = time * projected_width + channel * self.kernel;
                for tap in 0..self.kernel {
                    output[channel * raw_len + destination + tap] += projected[source + tap];
                }
            }
        }
        Ok((output, raw_len))
    }
}

#[derive(Debug)]
struct LstmLayer {
    weight_ih: Vec<f32>,
    weight_hh: Vec<f32>,
    bias_ih: Vec<f32>,
    bias_hh: Vec<f32>,
}

#[derive(Debug)]
struct Lstm {
    dimension: usize,
    layers: Vec<LstmLayer>,
}

impl Lstm {
    fn load(file: &GgufFile, prefix: &str, dimension: usize, layer_count: usize) -> Result<Self> {
        let gates = 4 * dimension;
        let mut layers = Vec::with_capacity(layer_count);
        for layer in 0..layer_count {
            layers.push(LstmLayer {
                weight_ih: tensor(
                    file,
                    &format!("{prefix}.weight_ih_l{layer}"),
                    &[gates, dimension],
                )?,
                weight_hh: tensor(
                    file,
                    &format!("{prefix}.weight_hh_l{layer}"),
                    &[gates, dimension],
                )?,
                bias_ih: tensor(file, &format!("{prefix}.bias_ih_l{layer}"), &[gates])?,
                bias_hh: tensor(file, &format!("{prefix}.bias_hh_l{layer}"), &[gates])?,
            });
        }
        Ok(Self { dimension, layers })
    }

    fn forward(&self, input: &[f32], time: usize, compute: &Compute) -> Result<Vec<f32>> {
        require_layout("LSTM input", input, self.dimension, time)?;
        let residual = input;
        let mut layer_input = input.to_vec();
        let gates = 4 * self.dimension;
        for layer in &self.layers {
            let mut output = vec![0.0f32; self.dimension * time];
            let mut hidden = vec![0.0f32; self.dimension];
            let mut cell = vec![0.0f32; self.dimension];
            let mut step_input = vec![0.0f32; self.dimension];
            let mut input_gates = vec![0.0f32; gates];
            let mut recurrent_gates = vec![0.0f32; gates];
            for step in 0..time {
                for dim in 0..self.dimension {
                    step_input[dim] = layer_input[dim * time + step];
                }
                compute.gemv_f32(
                    gates,
                    self.dimension,
                    &layer.weight_ih,
                    &step_input,
                    Some(&layer.bias_ih),
                    &mut input_gates,
                )?;
                compute.gemv_f32(
                    gates,
                    self.dimension,
                    &layer.weight_hh,
                    &hidden,
                    Some(&layer.bias_hh),
                    &mut recurrent_gates,
                )?;
                for dim in 0..self.dimension {
                    let input_gate = sigmoid(input_gates[dim] + recurrent_gates[dim]);
                    let forget_gate = sigmoid(
                        input_gates[self.dimension + dim] + recurrent_gates[self.dimension + dim],
                    );
                    let candidate = (input_gates[2 * self.dimension + dim]
                        + recurrent_gates[2 * self.dimension + dim])
                        .tanh();
                    let output_gate = sigmoid(
                        input_gates[3 * self.dimension + dim]
                            + recurrent_gates[3 * self.dimension + dim],
                    );
                    cell[dim] = forget_gate * cell[dim] + input_gate * candidate;
                    hidden[dim] = output_gate * cell[dim].tanh();
                    output[dim * time + step] = hidden[dim];
                }
            }
            layer_input = output;
        }
        for (value, &skip) in layer_input.iter_mut().zip(residual) {
            *value += skip;
        }
        Ok(layer_input)
    }
}

#[derive(Debug, Clone, Copy)]
enum WeightNames {
    Legacy,
    Parametrized,
}

impl WeightNames {
    fn names(self, prefix: &str) -> (String, String) {
        match self {
            Self::Legacy => (format!("{prefix}.weight_g"), format!("{prefix}.weight_v")),
            Self::Parametrized => (
                format!("{prefix}.parametrizations.weight.original0"),
                format!("{prefix}.parametrizations.weight.original1"),
            ),
        }
    }
}

fn reconstruct_weight_norm(
    g: &[f32],
    v: &[f32],
    primary: usize,
    plane: usize,
    name: &str,
) -> Result<Vec<f32>> {
    if g.len() != primary || v.len() != primary * plane {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: weight-norm `{name}` has g/v lengths {}/{}, expected {primary}/{}",
            g.len(),
            v.len(),
            primary * plane
        )));
    }
    let mut output = vec![0.0f32; v.len()];
    for row in 0..primary {
        let source = &v[row * plane..(row + 1) * plane];
        let norm = source.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: weight-norm `{name}` row {row} has invalid L2 norm {norm}"
            )));
        }
        let scale = g[row] / norm;
        for (destination, &value) in output[row * plane..(row + 1) * plane]
            .iter_mut()
            .zip(source)
        {
            *destination = value * scale;
        }
    }
    reject_non_finite(name, &output)?;
    Ok(output)
}

fn elu_inplace(values: &mut [f32]) {
    for value in values {
        if *value < 0.0 {
            *value = (*value).exp_m1();
        }
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn require_layout(label: &str, values: &[f32], channels: usize, time: usize) -> Result<()> {
    let expected = channels.checked_mul(time).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{ARCH}: {label} shape overflows usize"))
    })?;
    if values.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "{ARCH}: {label} has {} values, expected {channels}x{time}={expected}",
            values.len()
        )));
    }
    Ok(())
}

fn validate_pcm(pcm: &[f32], sample_rate: u32) -> Result<()> {
    if sample_rate != SAMPLE_RATE {
        return Err(VokraError::InvalidArgument(format!(
            "{ARCH}: expected {SAMPLE_RATE} Hz mono PCM, got {sample_rate} Hz; resample explicitly before AudioSeal"
        )));
    }
    if pcm.is_empty() {
        return Err(VokraError::InvalidArgument(format!("{ARCH}: PCM is empty")));
    }
    reject_non_finite("PCM", pcm)
}

fn validate_message(message: &[u8; NBITS]) -> Result<()> {
    if let Some((index, value)) = message
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| *value > 1)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{ARCH}: message bit {index} is {value}, expected 0 or 1"
        )));
    }
    Ok(())
}

fn validate_threshold(label: &str, value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(VokraError::InvalidArgument(format!(
            "{ARCH}: {label} must be finite and in [0, 1], got {value}"
        )));
    }
    Ok(())
}

fn reject_non_finite(label: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "{ARCH}: {label} contains non-finite value {value} at index {index}"
        )));
    }
    Ok(())
}

fn tensor(file: &GgufFile, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("{ARCH}: missing tensor `{name}`")))?;
    let expected = dims.iter().map(|&dim| dim as u64).collect::<Vec<_>>();
    if info.dimensions != expected {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: tensor `{name}` has shape {:?}, expected {expected:?}",
            info.dimensions
        )));
    }
    let values = file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("{ARCH}: reading tensor `{name}` failed: {error}"))
    })?;
    reject_non_finite(name, &values).map_err(|error| VokraError::ModelLoad(error.to_string()))?;
    Ok(values)
}

fn validate_contract_metadata(file: &GgufFile) -> Result<()> {
    require_string(file, KEY_CHECKPOINT_REVISION, CHECKPOINT_REVISION)?;
    require_string(file, KEY_SOURCE_REVISION, SOURCE_REVISION)?;
    for (key, expected) in [
        (KEY_SAMPLE_RATE, SAMPLE_RATE),
        (KEY_NBITS, NBITS as u32),
        (KEY_CHANNELS, 1),
        (KEY_DIMENSION, DIMENSION as u32),
        (KEY_N_FILTERS, N_FILTERS as u32),
        (KEY_N_RESIDUAL_LAYERS, 1),
        (KEY_COMPRESS, 2),
        (KEY_DILATION_BASE, 2),
        (KEY_KERNEL_SIZE, 7),
        (KEY_LAST_KERNEL_SIZE, 7),
        (KEY_RESIDUAL_KERNEL_SIZE, 3),
        (KEY_LSTM_LAYERS, 2),
        (KEY_DETECTOR_OUTPUT_DIM, DETECTOR_DIM as u32),
        (KEY_HOP_LENGTH, HOP_LENGTH as u32),
    ] {
        require_u32(file, key, expected)?;
    }
    let ratios = RATIOS.map(|ratio| ratio as u32);
    require_u32_array(file, KEY_RATIOS, &ratios)?;
    require_string(file, KEY_ACTIVATION, "elu")?;
    require_string(file, KEY_NORM, "weight_norm")?;
    require_string(file, KEY_PAD_MODE, "constant")?;
    require_bool(file, KEY_TRUE_SKIP, true)?;
    require_bool(file, KEY_BASE_CAUSAL, false)?;
    require_bool(file, KEY_STREAMING_CAUSAL, true)?;
    require_bool(file, KEY_NORMALIZER, false)?;
    Ok(())
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: metadata `{key}` is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_u64);
    if actual != Some(u64::from(expected)) {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: metadata `{key}` is {actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_bool);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: metadata `{key}` is {actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_u32_array(file: &GgufFile, key: &str, expected: &[u32]) -> Result<()> {
    let Some(array) = file.get(key).and_then(GgufMetadataValue::as_array) else {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: metadata `{key}` is missing or not an array"
        )));
    };
    if array.element_type != GgufValueType::U32 {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: metadata `{key}` has element type {:?}, expected U32",
            array.element_type
        )));
    }
    let actual = array
        .values
        .iter()
        .map(GgufMetadataValue::as_u64)
        .collect::<Option<Vec<_>>>();
    let expected = expected.iter().copied().map(u64::from).collect::<Vec<_>>();
    if actual.as_deref() != Some(expected.as_slice()) {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: metadata `{key}` is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn validate_manifest(file: &GgufFile) -> Result<()> {
    let expected = expected_manifest();
    if expected.len() != TENSOR_COUNT || file.tensors().len() != TENSOR_COUNT {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: checkpoint has {} tensors, expected exactly {TENSOR_COUNT}",
            file.tensors().len()
        )));
    }
    let mut seen = BTreeSet::new();
    for info in file.tensors() {
        let expected_shape = expected.get(&info.name).ok_or_else(|| {
            VokraError::ModelLoad(format!("{ARCH}: unexpected tensor `{}`", info.name))
        })?;
        if &info.dimensions != expected_shape {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: tensor `{}` has shape {:?}, expected {expected_shape:?}",
                info.name, info.dimensions
            )));
        }
        if info.dtype != GgmlType::F32 {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: tensor `{}` is {:?}, expected F32 for checkpoint {CHECKPOINT_REVISION}",
                info.name, info.dtype
            )));
        }
        seen.insert(info.name.as_str());
    }
    if let Some(missing) = expected.keys().find(|name| !seen.contains(name.as_str())) {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: missing tensor `{missing}`"
        )));
    }
    Ok(())
}

fn expected_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut out = BTreeMap::new();
    add_detector_manifest(&mut out, "detector_base", WeightNames::Legacy);
    add_detector_manifest(&mut out, "detector_streaming", WeightNames::Parametrized);
    add_generator_manifest(&mut out, "generator_base");
    add_generator_manifest(&mut out, "generator_streaming");
    debug_assert_eq!(out.len(), TENSOR_COUNT);
    out
}

fn add_generator_manifest(out: &mut BTreeMap<String, Vec<u64>>, root: &str) {
    add_decoder_manifest(out, &format!("{root}.decoder"), WeightNames::Legacy);
    add_encoder_manifest(out, &format!("{root}.encoder"), WeightNames::Legacy);
    insert_manifest(
        out,
        format!("{root}.msg_processor.msg_processor.weight"),
        &[32, 128],
    );
}

fn add_detector_manifest(out: &mut BTreeMap<String, Vec<u64>>, root: &str, names: WeightNames) {
    let encoder = format!("{root}.detector.0");
    add_encoder_manifest(out, &encoder, names);
    insert_manifest(
        out,
        format!("{encoder}.reverse_convolution.weight"),
        &[128, 32, 320],
    );
    insert_manifest(out, format!("{encoder}.reverse_convolution.bias"), &[32]);
    insert_manifest(out, format!("{root}.detector.1.weight"), &[18, 32, 1]);
    insert_manifest(out, format!("{root}.detector.1.bias"), &[18]);
}

fn add_encoder_manifest(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str, names: WeightNames) {
    add_conv_manifest(out, &format!("{prefix}.model.0.conv.conv"), 32, 1, 7, names);
    for (residual, down, channels, next, kernel) in [
        (1, 3, 32, 64, 4),
        (4, 6, 64, 128, 8),
        (7, 9, 128, 256, 10),
        (10, 12, 256, 512, 16),
    ] {
        add_residual_manifest(out, prefix, residual, channels, names);
        add_conv_manifest(
            out,
            &format!("{prefix}.model.{down}.conv.conv"),
            next,
            channels,
            kernel,
            names,
        );
    }
    add_lstm_manifest(out, &format!("{prefix}.model.13.lstm"));
    add_conv_manifest(
        out,
        &format!("{prefix}.model.15.conv.conv"),
        128,
        512,
        7,
        names,
    );
}

fn add_decoder_manifest(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str, names: WeightNames) {
    add_conv_manifest(
        out,
        &format!("{prefix}.model.0.conv.conv"),
        512,
        128,
        7,
        names,
    );
    add_lstm_manifest(out, &format!("{prefix}.model.1.lstm"));
    for (transpose, residual, channels, next, kernel) in [
        (3, 4, 512, 256, 16),
        (6, 7, 256, 128, 10),
        (9, 10, 128, 64, 8),
        (12, 13, 64, 32, 4),
    ] {
        add_conv_transpose_manifest(
            out,
            &format!("{prefix}.model.{transpose}.convtr.convtr"),
            channels,
            next,
            kernel,
            names,
        );
        add_residual_manifest(out, prefix, residual, next, names);
    }
    add_conv_manifest(
        out,
        &format!("{prefix}.model.15.conv.conv"),
        1,
        32,
        7,
        names,
    );
}

fn add_residual_manifest(
    out: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    index: usize,
    channels: u64,
    names: WeightNames,
) {
    let hidden = channels / 2;
    add_conv_manifest(
        out,
        &format!("{prefix}.model.{index}.block.1.conv.conv"),
        hidden,
        channels,
        3,
        names,
    );
    add_conv_manifest(
        out,
        &format!("{prefix}.model.{index}.block.3.conv.conv"),
        channels,
        hidden,
        1,
        names,
    );
}

fn add_conv_manifest(
    out: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    output: u64,
    input: u64,
    kernel: u64,
    names: WeightNames,
) {
    add_weight_norm_manifest(out, prefix, output, &[output, input, kernel], names);
    insert_manifest(out, format!("{prefix}.bias"), &[output]);
}

fn add_conv_transpose_manifest(
    out: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    input: u64,
    output: u64,
    kernel: u64,
    names: WeightNames,
) {
    add_weight_norm_manifest(out, prefix, input, &[input, output, kernel], names);
    insert_manifest(out, format!("{prefix}.bias"), &[output]);
}

fn add_weight_norm_manifest(
    out: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    primary: u64,
    weight_shape: &[u64],
    names: WeightNames,
) {
    let (g, v) = names.names(prefix);
    insert_manifest(out, g, &[primary, 1, 1]);
    insert_manifest(out, v, weight_shape);
}

fn add_lstm_manifest(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str) {
    for layer in 0..2 {
        insert_manifest(out, format!("{prefix}.weight_ih_l{layer}"), &[2048, 512]);
        insert_manifest(out, format!("{prefix}.weight_hh_l{layer}"), &[2048, 512]);
        insert_manifest(out, format!("{prefix}.bias_ih_l{layer}"), &[2048]);
        insert_manifest(out, format!("{prefix}.bias_hh_l{layer}"), &[2048]);
    }
}

fn insert_manifest(out: &mut BTreeMap<String, Vec<u64>>, name: String, shape: &[u64]) {
    assert!(out.insert(name.clone(), shape.to_vec()).is_none(), "{name}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
    ) -> Conv1d {
        Conv1d {
            input_channels,
            output_channels,
            kernel,
            stride,
            weight,
            bias,
        }
    }

    #[test]
    fn manifest_matches_the_public_four_variant_contract() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        for (prefix, count) in [
            ("generator_base.", 101),
            ("generator_streaming.", 101),
            ("detector_base.", 54),
            ("detector_streaming.", 54),
        ] {
            assert_eq!(
                manifest
                    .keys()
                    .filter(|name| name.starts_with(prefix))
                    .count(),
                count,
                "{prefix}"
            );
        }
    }

    #[test]
    fn weight_norm_reconstructs_each_primary_plane() {
        let g = [2.0, 3.0];
        let v = [3.0, 4.0, 0.0, 0.0, 0.0, 5.0];
        let actual = reconstruct_weight_norm(&g, &v, 2, 3, "test").unwrap();
        let expected = [1.2, 1.6, 0.0, 0.0, 0.0, 3.0];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn noncausal_and_causal_padding_match_upstream_length_rules() {
        let layer = conv(1, 1, 4, 2, vec![1.0; 4], vec![0.0]);
        let compute = Compute::cpu();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (base, base_len) = layer.forward(&input, input.len(), false, &compute).unwrap();
        let (causal, causal_len) = layer.forward(&input, input.len(), true, &compute).unwrap();
        assert_eq!(base_len, 3);
        assert_eq!(causal_len, 3);
        assert_eq!(base, vec![6.0, 14.0, 9.0]);
        assert_eq!(causal, vec![3.0, 10.0, 12.0]);
    }

    #[test]
    fn conv_transpose_gemm_scatter_and_trim_match_definition() {
        let layer = ConvTranspose1d {
            input_channels: 1,
            output_channels: 1,
            kernel: 4,
            stride: 2,
            weight: vec![1.0, 2.0, 3.0, 4.0],
            bias: vec![0.5],
        };
        let compute = Compute::cpu();
        let (raw, raw_len) = layer.forward_raw(&[1.0, 2.0], 2, &compute).unwrap();
        assert_eq!(raw_len, 6);
        assert_eq!(raw, vec![1.5, 2.5, 5.5, 8.5, 6.5, 8.5]);
        let (base, base_len) = layer.forward(&[1.0, 2.0], 2, false, &compute).unwrap();
        assert_eq!(base_len, 4);
        assert_eq!(base, vec![2.5, 5.5, 8.5, 6.5]);
        let (causal, causal_len) = layer.forward(&[1.0, 2.0], 2, true, &compute).unwrap();
        assert_eq!(causal_len, 4);
        assert_eq!(causal, vec![1.5, 2.5, 5.5, 8.5]);
    }

    #[test]
    fn invalid_message_and_thresholds_fail_loudly() {
        let mut message = [0u8; NBITS];
        message[7] = 2;
        assert!(validate_message(&message).is_err());
        assert!(validate_threshold("x", -0.1).is_err());
        assert!(validate_threshold("x", 1.1).is_err());
        assert!(validate_threshold("x", f32::NAN).is_err());
    }
}
