//! Native NVIDIA TitaNet-L speaker embedding.
//!
//! This module implements the complete inference graph from the pinned NeMo
//! 1.10.0 source and the public `speakerverification_en_titanet_large`
//! checkpoint: the 80-bin NeMo filterbank frontend, five depthwise-separable
//! Jasper/ContextNet blocks with squeeze-excitation, attentive statistics
//! pooling, and the 192-dimensional speaker projection. Every learned
//! convolution, SE projection, and attention softmax is dispatched through
//! [`Compute`]. Selecting Metal therefore executes the complete learned graph
//! on Metal and never falls back silently to CPU.

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::ir::graph::{Normalization, PadMode, StftAttrs, Window, WindowSymmetry};
use vokra_core::{BackendKind, LicenseClass, Result, SpeakerEngine, VokraError};
use vokra_ops::stft;

use crate::compute::{Compute, HotOp};

/// `vokra.model.arch` accepted by this runtime.
pub const ARCH: &str = "titanet-large";
/// Canonical public model identity.
pub const NAME: &str = "titanet-large";
/// Model task category.
pub const CATEGORY: &str = "speaker";
/// Audited upstream Hugging Face repository.
pub const UPSTREAM_HF: &str = "nvidia/speakerverification_en_titanet_large";
/// Immutable upstream checkpoint revision.
pub const UPSTREAM_REVISION: &str = "0dc382f40121a5fbd34db10a2bb04d826c2be6a8";
/// NeMo source revision corresponding to the checkpoint's declared 1.10.0 release.
pub const SOURCE_REVISION: &str = "082c5ae26168796d3ebac6adcf54bb8b5354daa1";
/// Output speaker-embedding width.
pub const EMBED_DIM: usize = 192;

const SAMPLE_RATE: u32 = 16_000;
const INPUT_DIM: usize = 80;
const FFT_SIZE: usize = 512;
const FFT_BINS: usize = FFT_SIZE / 2 + 1;
const WINDOW_SIZE: usize = 400;
const HOP_SIZE: usize = 160;
const PREEMPH: f32 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8;
const FRONTEND_STD_EPS: f32 = 1.0e-5;
const ENCODER_BN_EPS: f32 = 1.0e-3;
const DECODER_BN_EPS: f32 = 1.0e-5;
const STATS_EPS: f32 = 1.0e-10;
const ENCODER_CHANNELS: usize = 1_024;
const OUTPUT_CHANNELS: usize = 3_072;
const ATTENTION_CHANNELS: usize = 128;
const STATS_CHANNELS: usize = OUTPUT_CHANNELS * 2;
const CLASS_COUNT: usize = 16_681;
const TENSOR_COUNT: usize = 108;
const TITANET_HOT_OPS: &[HotOp] = &[
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::Gemv,
    HotOp::Softmax,
];

const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_SOURCE_REVISION: &str = "vokra.titanet.source_revision";
const KEY_SAMPLE_RATE: &str = "vokra.titanet.sample_rate";
const KEY_N_MELS: &str = "vokra.titanet.n_mels";
const KEY_N_FFT: &str = "vokra.titanet.n_fft";
const KEY_WIN_LENGTH: &str = "vokra.titanet.win_length";
const KEY_HOP_LENGTH: &str = "vokra.titanet.hop_length";
const KEY_EMBED_DIM: &str = "vokra.titanet.embed_dim";
const KEY_ENCODER_CHANNELS: &str = "vokra.titanet.encoder_channels";
const KEY_OUTPUT_CHANNELS: &str = "vokra.titanet.output_channels";
const KEY_ATTENTION_CHANNELS: &str = "vokra.titanet.attention_channels";
const KEY_ENCODER_BN_EPS: &str = "vokra.titanet.encoder_bn_eps";
const KEY_DECODER_BN_EPS: &str = "vokra.titanet.decoder_bn_eps";
const KEY_STATS_EPS: &str = "vokra.titanet.stats_eps";
const KEY_FRONTEND: &str = "vokra.titanet.frontend";
const KEY_BLOCKS: &str = "vokra.titanet.blocks";
const KEY_POOLING: &str = "vokra.titanet.pooling";
const KEY_LAYOUT: &str = "vokra.titanet.artifact_layout";
const CONTRACT_KEYS: [&str; 17] = [
    KEY_SOURCE_REVISION,
    KEY_SAMPLE_RATE,
    KEY_N_MELS,
    KEY_N_FFT,
    KEY_WIN_LENGTH,
    KEY_HOP_LENGTH,
    KEY_EMBED_DIM,
    KEY_ENCODER_CHANNELS,
    KEY_OUTPUT_CHANNELS,
    KEY_ATTENTION_CHANNELS,
    KEY_ENCODER_BN_EPS,
    KEY_DECODER_BN_EPS,
    KEY_STATS_EPS,
    KEY_FRONTEND,
    KEY_BLOCKS,
    KEY_POOLING,
    KEY_LAYOUT,
];

#[derive(Debug)]
struct FoldedBatchNorm {
    scale: Vec<f32>,
    shift: Vec<f32>,
}

impl FoldedBatchNorm {
    fn bind(file: &GgufFile, prefix: &str, channels: usize, eps: f32) -> Result<Self> {
        let gamma = tensor(file, &format!("{prefix}.weight"), &[channels])?;
        let beta = tensor(file, &format!("{prefix}.bias"), &[channels])?;
        let mean = tensor(file, &format!("{prefix}.running_mean"), &[channels])?;
        let variance = tensor(file, &format!("{prefix}.running_var"), &[channels])?;
        let mut scale = vec![0.0; channels];
        let mut shift = vec![0.0; channels];
        for channel in 0..channels {
            scale[channel] = gamma[channel] / (variance[channel] + eps).sqrt();
            shift[channel] = beta[channel] - mean[channel] * scale[channel];
        }
        Ok(Self { scale, shift })
    }

    fn apply(&self, values: &mut [f32], frames: usize) {
        debug_assert_eq!(values.len(), self.scale.len() * frames);
        for channel in 0..self.scale.len() {
            let scale = self.scale[channel];
            let shift = self.shift[channel];
            for value in &mut values[channel * frames..(channel + 1) * frames] {
                *value = *value * scale + shift;
            }
        }
    }
}

#[derive(Debug)]
struct SeparableUnit {
    depthwise: Vec<f32>,
    pointwise: Vec<f32>,
    norm: FoldedBatchNorm,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
}

impl SeparableUnit {
    fn bind(
        file: &GgufFile,
        base: &str,
        offset: usize,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
    ) -> Result<Self> {
        Ok(Self {
            depthwise: tensor(
                file,
                &format!("{base}.{offset}.conv.weight"),
                &[input_channels, 1, kernel],
            )?,
            pointwise: tensor(
                file,
                &format!("{base}.{}.conv.weight", offset + 1),
                &[output_channels, input_channels, 1],
            )?,
            norm: FoldedBatchNorm::bind(
                file,
                &format!("{base}.{}", offset + 2),
                output_channels,
                ENCODER_BN_EPS,
            )?,
            input_channels,
            output_channels,
            kernel,
        })
    }

    fn forward(&self, input: &[f32], frames: usize, compute: &Compute) -> Result<Vec<f32>> {
        if input.len() != self.input_channels * frames {
            return Err(VokraError::InvalidArgument(format!(
                "titanet: separable convolution input has {} values, expected {} x {frames}",
                input.len(),
                self.input_channels
            )));
        }
        let mut depthwise = vec![0.0; self.input_channels * frames];
        compute.grouped_conv1d_f32(
            input,
            self.input_channels,
            frames,
            &self.depthwise,
            self.input_channels,
            self.kernel,
            None,
            1,
            self.kernel / 2,
            self.input_channels,
            &mut depthwise,
        )?;
        let mut output = vec![0.0; self.output_channels * frames];
        compute.conv1d_f32(
            &depthwise,
            self.input_channels,
            frames,
            &self.pointwise,
            self.output_channels,
            1,
            None,
            1,
            0,
            &mut output,
        )?;
        self.norm.apply(&mut output, frames);
        Ok(output)
    }
}

#[derive(Debug)]
struct SqueezeExcite {
    reduce: Vec<f32>,
    expand: Vec<f32>,
    channels: usize,
    hidden: usize,
}

impl SqueezeExcite {
    fn bind(file: &GgufFile, prefix: &str, channels: usize) -> Result<Self> {
        let hidden = channels / 8;
        Ok(Self {
            reduce: tensor(file, &format!("{prefix}.fc.0.weight"), &[hidden, channels])?,
            expand: tensor(file, &format!("{prefix}.fc.2.weight"), &[channels, hidden])?,
            channels,
            hidden,
        })
    }

    fn apply(&self, values: &mut [f32], frames: usize, compute: &Compute) -> Result<()> {
        debug_assert_eq!(values.len(), self.channels * frames);
        let mut mean = vec![0.0; self.channels];
        for (channel, row) in values.chunks_exact(frames).enumerate() {
            mean[channel] = row.iter().copied().sum::<f32>() / frames as f32;
        }
        let mut hidden = vec![0.0; self.hidden];
        compute.gemv_f32(
            self.hidden,
            self.channels,
            &self.reduce,
            &mean,
            None,
            &mut hidden,
        )?;
        relu(&mut hidden);
        let mut scale = vec![0.0; self.channels];
        compute.gemv_f32(
            self.channels,
            self.hidden,
            &self.expand,
            &hidden,
            None,
            &mut scale,
        )?;
        for value in &mut scale {
            *value = 1.0 / (1.0 + (-*value).exp());
        }
        for channel in 0..self.channels {
            let gain = scale[channel];
            for value in &mut values[channel * frames..(channel + 1) * frames] {
                *value *= gain;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct ResidualProjection {
    weight: Vec<f32>,
    norm: FoldedBatchNorm,
    input_channels: usize,
    output_channels: usize,
}

impl ResidualProjection {
    fn bind(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: tensor(
                file,
                &format!("{prefix}.0.conv.weight"),
                &[output_channels, input_channels, 1],
            )?,
            norm: FoldedBatchNorm::bind(
                file,
                &format!("{prefix}.1"),
                output_channels,
                ENCODER_BN_EPS,
            )?,
            input_channels,
            output_channels,
        })
    }

    fn forward(&self, input: &[f32], frames: usize, compute: &Compute) -> Result<Vec<f32>> {
        let mut output = vec![0.0; self.output_channels * frames];
        compute.conv1d_f32(
            input,
            self.input_channels,
            frames,
            &self.weight,
            self.output_channels,
            1,
            None,
            1,
            0,
            &mut output,
        )?;
        self.norm.apply(&mut output, frames);
        Ok(output)
    }
}

#[derive(Debug)]
struct JasperBlock {
    units: Vec<SeparableUnit>,
    se: SqueezeExcite,
    residual: Option<ResidualProjection>,
}

impl JasperBlock {
    fn bind(
        file: &GgufFile,
        block: usize,
        input_channels: usize,
        output_channels: usize,
        repeat: usize,
        kernel: usize,
        residual: bool,
    ) -> Result<Self> {
        let base = format!("encoder.encoder.{block}");
        let mut units = Vec::with_capacity(repeat);
        let mut channels = input_channels;
        for unit in 0..repeat {
            let offset = unit * 5;
            units.push(SeparableUnit::bind(
                file,
                &format!("{base}.mconv"),
                offset,
                channels,
                output_channels,
                kernel,
            )?);
            channels = output_channels;
        }
        let se_index = if repeat == 1 { 3 } else { 13 };
        let residual = residual
            .then(|| {
                ResidualProjection::bind(
                    file,
                    &format!("{base}.res.0"),
                    input_channels,
                    output_channels,
                )
            })
            .transpose()?;
        Ok(Self {
            units,
            se: SqueezeExcite::bind(file, &format!("{base}.mconv.{se_index}"), output_channels)?,
            residual,
        })
    }

    fn forward(&self, input: &[f32], frames: usize, compute: &Compute) -> Result<Vec<f32>> {
        let mut output = input.to_vec();
        for (index, unit) in self.units.iter().enumerate() {
            output = unit.forward(&output, frames, compute)?;
            if index + 1 != self.units.len() {
                relu(&mut output);
            }
        }
        self.se.apply(&mut output, frames, compute)?;
        if let Some(residual) = &self.residual {
            let residual = residual.forward(input, frames, compute)?;
            add_inplace(&mut output, &residual);
        }
        relu(&mut output);
        Ok(output)
    }
}

#[derive(Debug)]
struct AttentionPool {
    first_weight: Vec<f32>,
    first_bias: Vec<f32>,
    first_norm: FoldedBatchNorm,
    second_weight: Vec<f32>,
    second_bias: Vec<f32>,
}

impl AttentionPool {
    fn bind(file: &GgufFile) -> Result<Self> {
        let prefix = "decoder._pooling.attention_layer";
        Ok(Self {
            first_weight: tensor(
                file,
                &format!("{prefix}.0.conv_layer.weight"),
                &[ATTENTION_CHANNELS, OUTPUT_CHANNELS * 3, 1],
            )?,
            first_bias: tensor(
                file,
                &format!("{prefix}.0.conv_layer.bias"),
                &[ATTENTION_CHANNELS],
            )?,
            first_norm: FoldedBatchNorm::bind(
                file,
                &format!("{prefix}.0.bn"),
                ATTENTION_CHANNELS,
                DECODER_BN_EPS,
            )?,
            second_weight: tensor(
                file,
                &format!("{prefix}.2.weight"),
                &[OUTPUT_CHANNELS, ATTENTION_CHANNELS, 1],
            )?,
            second_bias: tensor(file, &format!("{prefix}.2.bias"), &[OUTPUT_CHANNELS])?,
        })
    }

    fn forward(&self, input: &[f32], frames: usize, compute: &Compute) -> Result<Vec<f32>> {
        if input.len() != OUTPUT_CHANNELS * frames || frames == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "titanet: attention pool expected {OUTPUT_CHANNELS} channels and at least one frame, got {} values / {frames} frames",
                input.len()
            )));
        }
        let (mean, std) = statistics(input, OUTPUT_CHANNELS, frames, STATS_EPS)?;
        let mut context = vec![0.0; OUTPUT_CHANNELS * 3 * frames];
        context[..input.len()].copy_from_slice(input);
        for channel in 0..OUTPUT_CHANNELS {
            let mean_row = (OUTPUT_CHANNELS + channel) * frames;
            let std_row = (OUTPUT_CHANNELS * 2 + channel) * frames;
            context[mean_row..mean_row + frames].fill(mean[channel]);
            context[std_row..std_row + frames].fill(std[channel]);
        }

        let mut attention = vec![0.0; ATTENTION_CHANNELS * frames];
        compute.conv1d_f32(
            &context,
            OUTPUT_CHANNELS * 3,
            frames,
            &self.first_weight,
            ATTENTION_CHANNELS,
            1,
            Some(&self.first_bias),
            1,
            0,
            &mut attention,
        )?;
        relu(&mut attention);
        self.first_norm.apply(&mut attention, frames);
        for value in &mut attention {
            *value = value.tanh();
        }
        let mut logits = vec![0.0; OUTPUT_CHANNELS * frames];
        compute.conv1d_f32(
            &attention,
            ATTENTION_CHANNELS,
            frames,
            &self.second_weight,
            OUTPUT_CHANNELS,
            1,
            Some(&self.second_bias),
            1,
            0,
            &mut logits,
        )?;
        let mut alpha = vec![0.0; logits.len()];
        compute.softmax_f32(&logits, &mut alpha, OUTPUT_CHANNELS, frames)?;
        weighted_statistics(input, &alpha, OUTPUT_CHANNELS, frames, STATS_EPS)
    }
}

#[derive(Debug)]
struct TitaNetWeights {
    mel_filterbank: Vec<f32>,
    stored_window: Vec<f32>,
    blocks: Vec<JasperBlock>,
    pooling: AttentionPool,
    embedding_norm: FoldedBatchNorm,
    embedding_weight: Vec<f32>,
    embedding_bias: Vec<f32>,
    #[allow(dead_code)]
    classifier_weight: Vec<f32>,
}

impl TitaNetWeights {
    fn bind(file: &GgufFile) -> Result<Self> {
        verify_manifest(file)?;
        let blocks = vec![
            JasperBlock::bind(file, 0, INPUT_DIM, ENCODER_CHANNELS, 1, 3, false)?,
            JasperBlock::bind(file, 1, ENCODER_CHANNELS, ENCODER_CHANNELS, 3, 7, true)?,
            JasperBlock::bind(file, 2, ENCODER_CHANNELS, ENCODER_CHANNELS, 3, 11, true)?,
            JasperBlock::bind(file, 3, ENCODER_CHANNELS, ENCODER_CHANNELS, 3, 15, true)?,
            JasperBlock::bind(file, 4, ENCODER_CHANNELS, OUTPUT_CHANNELS, 1, 1, false)?,
        ];
        Ok(Self {
            mel_filterbank: tensor(
                file,
                "preprocessor.featurizer.fb",
                &[1, INPUT_DIM, FFT_BINS],
            )?,
            stored_window: tensor(file, "preprocessor.featurizer.window", &[WINDOW_SIZE])?,
            blocks,
            pooling: AttentionPool::bind(file)?,
            embedding_norm: FoldedBatchNorm::bind(
                file,
                "decoder.emb_layers.0.0",
                STATS_CHANNELS,
                DECODER_BN_EPS,
            )?,
            embedding_weight: tensor(
                file,
                "decoder.emb_layers.0.1.weight",
                &[EMBED_DIM, STATS_CHANNELS, 1],
            )?,
            embedding_bias: tensor(file, "decoder.emb_layers.0.1.bias", &[EMBED_DIM])?,
            classifier_weight: tensor(file, "decoder.final.weight", &[CLASS_COUNT, EMBED_DIM])?,
        })
    }

    fn encode_features(
        &self,
        features: &[f32],
        frames: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if frames == 0 || features.len() != INPUT_DIM * frames {
            return Err(VokraError::InvalidArgument(format!(
                "titanet: expected channel-major [80, frames] features, got {} values / {frames} frames",
                features.len()
            )));
        }
        let mut encoded = features.to_vec();
        for block in &self.blocks {
            encoded = block.forward(&encoded, frames, compute)?;
        }
        let mut pooled = self.pooling.forward(&encoded, frames, compute)?;
        self.embedding_norm.apply(&mut pooled, 1);
        let mut embedding = vec![0.0; EMBED_DIM];
        compute.conv1d_f32(
            &pooled,
            STATS_CHANNELS,
            1,
            &self.embedding_weight,
            EMBED_DIM,
            1,
            Some(&self.embedding_bias),
            1,
            0,
            &mut embedding,
        )?;
        Ok(embedding)
    }

    fn frontend(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        if pcm.len() < 2 {
            return Err(VokraError::InvalidArgument(
                "titanet: PCM must contain at least two samples for reflect padding".to_owned(),
            ));
        }
        validate_stored_window(&self.stored_window)?;
        let mut emphasized = Vec::with_capacity(pcm.len());
        emphasized.push(pcm[0]);
        for pair in pcm.windows(2) {
            emphasized.push(pair[1] - PREEMPH * pair[0]);
        }
        let attrs = StftAttrs {
            n_fft: FFT_SIZE,
            hop_length: HOP_SIZE,
            win_length: WINDOW_SIZE,
            window: Window::Hann,
            window_symmetry: WindowSymmetry::Symmetric,
            center: true,
            pad_mode: PadMode::Reflect,
            normalization: Normalization::Backward,
            causal: false,
            real_input: true,
        };
        let spectrogram = stft(&emphasized, &attrs)?;
        let frames = spectrogram.frames;
        if frames < 2 {
            return Err(VokraError::InvalidArgument(format!(
                "titanet: frontend produced {frames} frame(s); at least two are required for NeMo per-feature normalization"
            )));
        }
        let power = spectrogram.power();
        let mut features = vec![0.0; INPUT_DIM * frames];
        for mel in 0..INPUT_DIM {
            let filter = &self.mel_filterbank[mel * FFT_BINS..(mel + 1) * FFT_BINS];
            for frame in 0..frames {
                let spectrum = &power[frame * FFT_BINS..(frame + 1) * FFT_BINS];
                let energy = filter
                    .iter()
                    .zip(spectrum)
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>();
                features[mel * frames + frame] = (energy + LOG_GUARD).ln();
            }
        }
        for row in features.chunks_exact_mut(frames) {
            let mean = row.iter().copied().sum::<f32>() / frames as f32;
            let variance = row
                .iter()
                .map(|value| {
                    let centered = value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / (frames - 1) as f32;
            let denominator = variance.sqrt() + FRONTEND_STD_EPS;
            for value in row {
                *value = (*value - mean) / denominator;
            }
        }
        Ok((features, frames))
    }
}

/// Complete native TitaNet-L inference handle.
#[derive(Debug)]
pub struct TitaNet {
    weights: TitaNetWeights,
    backend: BackendKind,
}

impl TitaNet {
    /// Strictly binds the exact public 108-tensor NVIDIA checkpoint.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        require_string(file, chunks::KEY_MODEL_NAME, NAME)?;
        require_string(file, KEY_CATEGORY, CATEGORY)?;
        require_string(file, KEY_UPSTREAM_HF, UPSTREAM_HF)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "cc-by-4.0")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::AttributionRequired.as_str(),
        )?;
        if file
            .get(chunks::KEY_PROVENANCE_ATTRIBUTION)
            .and_then(|value| value.as_str())
            .filter(|value| !value.trim().is_empty())
            .is_none()
        {
            return Err(VokraError::ModelLoad(format!(
                "titanet: `{}` is missing or empty for the CC-BY-4.0 checkpoint",
                chunks::KEY_PROVENANCE_ATTRIBUTION
            )));
        }
        let weights = TitaNetWeights::bind(file)?;
        verify_optional_contract(file)?;
        Ok(Self {
            weights,
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and strictly binds a GGUF file.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Selects one backend for the complete learned graph.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the selected backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the normalized provenance license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        LicenseClass::AttributionRequired
    }

    /// Runs the NeMo PCM frontend and complete TitaNet speaker network.
    pub fn embed_pcm(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        if sample_rate != SAMPLE_RATE {
            return Err(VokraError::InvalidArgument(format!(
                "titanet: expected {SAMPLE_RATE} Hz mono PCM, got {sample_rate} Hz; resample offline first"
            )));
        }
        let (features, frames) = self.weights.frontend(pcm)?;
        self.embed_features(&features, frames)
    }

    /// Runs only the learned network on channel-major `[80, frames]` features.
    pub fn embed_features(&self, features: &[f32], frames: usize) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, TITANET_HOT_OPS)?;
        self.weights.encode_features(features, frames, &compute)
    }

    /// Computes the pinned NeMo frontend for parity checks.
    pub fn frontend_features(&self, pcm: &[f32], sample_rate: u32) -> Result<(Vec<f32>, usize)> {
        if sample_rate != SAMPLE_RATE {
            return Err(VokraError::InvalidArgument(format!(
                "titanet: expected {SAMPLE_RATE} Hz mono PCM, got {sample_rate} Hz"
            )));
        }
        self.weights.frontend(pcm)
    }
}

impl SpeakerEngine for TitaNet {
    fn embed(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        self.embed_pcm(pcm, sample_rate)
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }
}

fn relu(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

fn add_inplace(destination: &mut [f32], source: &[f32]) {
    debug_assert_eq!(destination.len(), source.len());
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += source;
    }
}

fn statistics(
    input: &[f32],
    channels: usize,
    frames: usize,
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if frames == 0 || input.len() != channels * frames {
        return Err(VokraError::InvalidArgument(format!(
            "titanet: statistics expected {channels} x {frames}, got {} values",
            input.len()
        )));
    }
    let mut mean = vec![0.0; channels];
    let mut std = vec![0.0; channels];
    for (channel, row) in input.chunks_exact(frames).enumerate() {
        mean[channel] = row.iter().copied().sum::<f32>() / frames as f32;
        let variance = row
            .iter()
            .map(|value| {
                let centered = value - mean[channel];
                centered * centered
            })
            .sum::<f32>()
            / frames as f32;
        std[channel] = variance.max(eps).sqrt();
    }
    Ok((mean, std))
}

fn weighted_statistics(
    input: &[f32],
    weights: &[f32],
    channels: usize,
    frames: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if input.len() != channels * frames || weights.len() != input.len() {
        return Err(VokraError::InvalidArgument(
            "titanet: weighted-statistics shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0; channels * 2];
    for channel in 0..channels {
        let row = &input[channel * frames..(channel + 1) * frames];
        let alpha = &weights[channel * frames..(channel + 1) * frames];
        let mean = row
            .iter()
            .zip(alpha)
            .map(|(value, weight)| value * weight)
            .sum::<f32>();
        let variance = row
            .iter()
            .zip(alpha)
            .map(|(value, weight)| {
                let centered = value - mean;
                weight * centered * centered
            })
            .sum::<f32>();
        output[channel] = mean;
        output[channels + channel] = variance.max(eps).sqrt();
    }
    Ok(output)
}

fn validate_stored_window(window: &[f32]) -> Result<()> {
    if window.len() != WINDOW_SIZE {
        return Err(VokraError::ModelLoad(format!(
            "titanet: stored Hann window has {} values, expected {WINDOW_SIZE}",
            window.len()
        )));
    }
    for (index, &actual) in window.iter().enumerate() {
        let expected = 0.5
            - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / (WINDOW_SIZE - 1) as f32).cos();
        if (actual - expected).abs() > 2.0e-6 {
            return Err(VokraError::ModelLoad(format!(
                "titanet: stored symmetric Hann window differs at index {index}: {actual} vs {expected}"
            )));
        }
    }
    Ok(())
}

fn expected_manifest() -> Vec<(String, Vec<usize>)> {
    let mut expected = Vec::with_capacity(TENSOR_COUNT);
    expected.push((
        "preprocessor.featurizer.fb".to_owned(),
        vec![1, INPUT_DIM, FFT_BINS],
    ));
    expected.push((
        "preprocessor.featurizer.window".to_owned(),
        vec![WINDOW_SIZE],
    ));

    push_jasper(&mut expected, 0, INPUT_DIM, ENCODER_CHANNELS, 1, 3, false);
    push_jasper(
        &mut expected,
        1,
        ENCODER_CHANNELS,
        ENCODER_CHANNELS,
        3,
        7,
        true,
    );
    push_jasper(
        &mut expected,
        2,
        ENCODER_CHANNELS,
        ENCODER_CHANNELS,
        3,
        11,
        true,
    );
    push_jasper(
        &mut expected,
        3,
        ENCODER_CHANNELS,
        ENCODER_CHANNELS,
        3,
        15,
        true,
    );
    push_jasper(
        &mut expected,
        4,
        ENCODER_CHANNELS,
        OUTPUT_CHANNELS,
        1,
        1,
        false,
    );

    let attention = "decoder._pooling.attention_layer";
    expected.push((
        format!("{attention}.0.conv_layer.weight"),
        vec![ATTENTION_CHANNELS, OUTPUT_CHANNELS * 3, 1],
    ));
    expected.push((
        format!("{attention}.0.conv_layer.bias"),
        vec![ATTENTION_CHANNELS],
    ));
    push_norm(
        &mut expected,
        &format!("{attention}.0.bn"),
        ATTENTION_CHANNELS,
    );
    expected.push((
        format!("{attention}.2.weight"),
        vec![OUTPUT_CHANNELS, ATTENTION_CHANNELS, 1],
    ));
    expected.push((format!("{attention}.2.bias"), vec![OUTPUT_CHANNELS]));
    push_norm(&mut expected, "decoder.emb_layers.0.0", STATS_CHANNELS);
    expected.push((
        "decoder.emb_layers.0.1.weight".to_owned(),
        vec![EMBED_DIM, STATS_CHANNELS, 1],
    ));
    expected.push(("decoder.emb_layers.0.1.bias".to_owned(), vec![EMBED_DIM]));
    expected.push((
        "decoder.final.weight".to_owned(),
        vec![CLASS_COUNT, EMBED_DIM],
    ));
    expected
}

fn push_jasper(
    expected: &mut Vec<(String, Vec<usize>)>,
    block: usize,
    input_channels: usize,
    output_channels: usize,
    repeat: usize,
    kernel: usize,
    residual: bool,
) {
    let base = format!("encoder.encoder.{block}");
    let mut channels = input_channels;
    for unit in 0..repeat {
        let offset = unit * 5;
        expected.push((
            format!("{base}.mconv.{offset}.conv.weight"),
            vec![channels, 1, kernel],
        ));
        expected.push((
            format!("{base}.mconv.{}.conv.weight", offset + 1),
            vec![output_channels, channels, 1],
        ));
        push_norm(
            expected,
            &format!("{base}.mconv.{}", offset + 2),
            output_channels,
        );
        channels = output_channels;
    }
    let se_index = if repeat == 1 { 3 } else { 13 };
    expected.push((
        format!("{base}.mconv.{se_index}.fc.0.weight"),
        vec![output_channels / 8, output_channels],
    ));
    expected.push((
        format!("{base}.mconv.{se_index}.fc.2.weight"),
        vec![output_channels, output_channels / 8],
    ));
    if residual {
        expected.push((
            format!("{base}.res.0.0.conv.weight"),
            vec![output_channels, input_channels, 1],
        ));
        push_norm(expected, &format!("{base}.res.0.1"), output_channels);
    }
}

fn push_norm(expected: &mut Vec<(String, Vec<usize>)>, prefix: &str, channels: usize) {
    for suffix in ["weight", "bias", "running_mean", "running_var"] {
        expected.push((format!("{prefix}.{suffix}"), vec![channels]));
    }
}

fn verify_manifest(file: &GgufFile) -> Result<()> {
    if file.tensors().len() != TENSOR_COUNT {
        return Err(VokraError::ModelLoad(format!(
            "titanet: tensor count is {}, expected exactly {TENSOR_COUNT}",
            file.tensors().len()
        )));
    }
    let expected = expected_manifest();
    debug_assert_eq!(expected.len(), TENSOR_COUNT);
    for (name, dimensions) in expected {
        check_dims(file, &name, &dimensions)?;
    }
    Ok(())
}

fn check_dims(file: &GgufFile, name: &str, expected: &[usize]) -> Result<()> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("titanet: missing tensor `{name}`")))?;
    let expected = expected
        .iter()
        .map(|&dimension| dimension as u64)
        .collect::<Vec<_>>();
    if info.dimensions != expected {
        return Err(VokraError::ModelLoad(format!(
            "titanet: tensor `{name}` has dims {:?}, expected {expected:?}",
            info.dimensions
        )));
    }
    Ok(())
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    check_dims(file, name, expected)?;
    file.tensor_f32(name)
        .map_err(|error| VokraError::ModelLoad(format!("titanet: reading `{name}`: {error}")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| VokraError::ModelLoad(format!("titanet: missing string `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "titanet: `{key}` is `{actual}`, expected `{expected}`"
        )));
    }
    Ok(())
}

fn verify_optional_contract(file: &GgufFile) -> Result<()> {
    if !CONTRACT_KEYS.iter().any(|key| file.get(key).is_some()) {
        return Ok(());
    }
    require_string(file, KEY_UPSTREAM_REVISION, UPSTREAM_REVISION)?;
    require_string(file, KEY_SOURCE_REVISION, SOURCE_REVISION)?;
    for (key, expected) in [
        (KEY_SAMPLE_RATE, SAMPLE_RATE),
        (KEY_N_MELS, INPUT_DIM as u32),
        (KEY_N_FFT, FFT_SIZE as u32),
        (KEY_WIN_LENGTH, WINDOW_SIZE as u32),
        (KEY_HOP_LENGTH, HOP_SIZE as u32),
        (KEY_EMBED_DIM, EMBED_DIM as u32),
        (KEY_ENCODER_CHANNELS, ENCODER_CHANNELS as u32),
        (KEY_OUTPUT_CHANNELS, OUTPUT_CHANNELS as u32),
        (KEY_ATTENTION_CHANNELS, ATTENTION_CHANNELS as u32),
    ] {
        require_u32(file, key, expected)?;
    }
    require_f32(file, KEY_ENCODER_BN_EPS, ENCODER_BN_EPS)?;
    require_f32(file, KEY_DECODER_BN_EPS, DECODER_BN_EPS)?;
    require_f32(file, KEY_STATS_EPS, STATS_EPS)?;
    require_string(file, KEY_FRONTEND, "nemo-filterbank-v1.10.0")?;
    require_string(file, KEY_BLOCKS, "1x3,3x7,3x11,3x15,1x1")?;
    require_string(file, KEY_POOLING, "attentive-statistics-population-v1")?;
    require_string(file, KEY_LAYOUT, "nemo-inference-108-v1")
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(|value| value.as_u64())
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| VokraError::ModelLoad(format!("titanet: missing u32 `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "titanet: `{key}` is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_f32(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    let actual = match file.get(key) {
        Some(GgufMetadataValue::F32(value)) => *value,
        _ => {
            return Err(VokraError::ModelLoad(format!(
                "titanet: missing f32 `{key}`"
            )));
        }
    };
    if actual.to_bits() != expected.to_bits() {
        return Err(VokraError::ModelLoad(format!(
            "titanet: `{key}` is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_manifest_has_108_tensors() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        let mut names = manifest.iter().map(|(name, _)| name).collect::<Vec<_>>();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), TENSOR_COUNT);
    }

    #[test]
    fn population_statistics_match_hand_calculation() {
        let (mean, std) = statistics(&[1.0, 3.0, 5.0, 7.0], 2, 2, STATS_EPS).unwrap();
        assert_eq!(mean, vec![2.0, 6.0]);
        assert_eq!(std, vec![1.0, 1.0]);
    }

    #[test]
    fn weighted_statistics_are_channel_wise() {
        let got = weighted_statistics(
            &[1.0, 3.0, 2.0, 6.0],
            &[0.25, 0.75, 0.5, 0.5],
            2,
            2,
            STATS_EPS,
        )
        .unwrap();
        assert!((got[0] - 2.5).abs() < 1.0e-6);
        assert!((got[1] - 4.0).abs() < 1.0e-6);
        assert!((got[2] - 0.75_f32.sqrt()).abs() < 1.0e-6);
        assert!((got[3] - 4.0_f32.sqrt()).abs() < 1.0e-6);
    }

    #[test]
    fn frontend_contract_is_pinned() {
        assert_eq!(SAMPLE_RATE, 16_000);
        assert_eq!((FFT_SIZE, WINDOW_SIZE, HOP_SIZE), (512, 400, 160));
        assert_eq!(INPUT_DIM, 80);
        assert_eq!(EMBED_DIM, 192);
        assert_eq!(SOURCE_REVISION.len(), 40);
        assert_eq!(UPSTREAM_REVISION.len(), 40);
    }
}
