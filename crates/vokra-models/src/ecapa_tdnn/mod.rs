//! Native SpeechBrain ECAPA-TDNN speaker embedding.
//!
//! This is the complete inference graph released as
//! `speechbrain/spkrec-ecapa-voxceleb`: the 80-bin SpeechBrain frontend,
//! reflect-padded TDNN stem, three SE-Res2Net blocks, multi-layer feature
//! aggregation, attentive statistics pooling and the 192-dimensional speaker
//! projection.  Learned convolutions and the channel-wise attention softmax
//! are dispatched through [`Compute`], so selecting Metal is observable and
//! never falls back silently to the CPU.

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{BackendKind, LicenseClass, Result, SpeakerEngine, VokraError};
use vokra_ops::{SpeechbrainFbankAttrs, speechbrain_fbank};

use crate::compute::{Compute, HotOp};

/// `vokra.model.arch` accepted by this runtime.
pub const ARCH: &str = "ecapa_tdnn";
/// Canonical public model identity.
pub const NAME: &str = "spkrec-ecapa-voxceleb";
/// Model task category.
pub const CATEGORY: &str = "speaker";
/// Pinned upstream Hugging Face repository.
pub const UPSTREAM_HF: &str = "speechbrain/spkrec-ecapa-voxceleb";
/// Audited upstream repository revision used by parity and future conversion.
pub const UPSTREAM_REVISION: &str = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286";
/// Output speaker-embedding width.
pub const EMBED_DIM: usize = 192;

const INPUT_DIM: usize = 80;
const TDNN_CHANNELS: usize = 1_024;
const RES2NET_SCALE: usize = 8;
const MFA_CHANNELS: usize = 3_072;
const ATTENTION_CHANNELS: usize = 128;
const BN_EPS: f32 = 1.0e-5;
const STATS_EPS: f32 = 1.0e-12;
const TENSOR_COUNT: usize = 200;
pub(crate) const ECAPA_HOT_OPS: &[HotOp] = &[HotOp::Conv1d, HotOp::Softmax];

/// Exact ECAPA topology needed by the shared speaker/lang-ID backbone.
///
/// This stays crate-private: public model handles expose task-specific
/// constructors, while both binders share one strict implementation.
#[derive(Debug, Clone)]
pub(crate) struct EcapaBackboneConfig {
    pub input_dim: usize,
    pub tdnn_channels: usize,
    pub res2net_scale: usize,
    pub mfa_channels: usize,
    pub attention_channels: usize,
    pub embedding_dim: usize,
    pub block_kernels: [usize; 3],
    pub block_dilations: [usize; 3],
    pub bn_eps: f32,
    pub stats_eps: f32,
    pub tensor_prefix: &'static str,
    pub diagnostic: &'static str,
}

impl EcapaBackboneConfig {
    fn speaker() -> Self {
        Self {
            input_dim: INPUT_DIM,
            tdnn_channels: TDNN_CHANNELS,
            res2net_scale: RES2NET_SCALE,
            mfa_channels: MFA_CHANNELS,
            attention_channels: ATTENTION_CHANNELS,
            embedding_dim: EMBED_DIM,
            block_kernels: [3, 3, 3],
            block_dilations: [2, 3, 4],
            bn_eps: BN_EPS,
            stats_eps: STATS_EPS,
            tensor_prefix: "",
            diagnostic: ARCH,
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("input_dim", self.input_dim),
            ("tdnn_channels", self.tdnn_channels),
            ("res2net_scale", self.res2net_scale),
            ("mfa_channels", self.mfa_channels),
            ("attention_channels", self.attention_channels),
            ("embedding_dim", self.embedding_dim),
        ] {
            if value == 0 {
                return Err(VokraError::ModelLoad(format!(
                    "{}: ECAPA config `{name}` must be non-zero",
                    self.diagnostic
                )));
            }
        }
        if self.res2net_scale < 2 || self.tdnn_channels % self.res2net_scale != 0 {
            return Err(VokraError::ModelLoad(format!(
                "{}: tdnn_channels={} is not divisible by res2net_scale={} >= 2",
                self.diagnostic, self.tdnn_channels, self.res2net_scale
            )));
        }
        if self.mfa_channels != self.tdnn_channels * 3 {
            return Err(VokraError::ModelLoad(format!(
                "{}: mfa_channels={} must equal 3 * tdnn_channels={}",
                self.diagnostic, self.mfa_channels, self.tdnn_channels
            )));
        }
        if self
            .block_kernels
            .iter()
            .any(|&value| value == 0 || value % 2 == 0)
            || self.block_dilations.contains(&0)
        {
            return Err(VokraError::ModelLoad(format!(
                "{}: ECAPA block kernels must be positive odd values and dilations must be positive",
                self.diagnostic
            )));
        }
        for (name, value) in [("bn_eps", self.bn_eps), ("stats_eps", self.stats_eps)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(VokraError::ModelLoad(format!(
                    "{}: ECAPA `{name}` must be finite and positive",
                    self.diagnostic
                )));
            }
        }
        Ok(())
    }

    fn tensor_name(&self, relative: &str) -> String {
        format!("{}{relative}", self.tensor_prefix)
    }

    fn max_reflect_padding(&self) -> usize {
        self.block_kernels
            .iter()
            .zip(self.block_dilations)
            .map(|(&kernel, dilation)| (kernel - 1) * dilation / 2)
            .chain(std::iter::once(2))
            .max()
            .unwrap_or(2)
    }
}

const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_SAMPLE_RATE: &str = "vokra.ecapa.sample_rate";
const KEY_N_MELS: &str = "vokra.ecapa.n_mels";
const KEY_N_FFT: &str = "vokra.ecapa.n_fft";
const KEY_WIN_LENGTH: &str = "vokra.ecapa.win_length";
const KEY_HOP_LENGTH: &str = "vokra.ecapa.hop_length";
const KEY_EMBED_DIM: &str = "vokra.ecapa.embed_dim";
const KEY_TDNN_CHANNELS: &str = "vokra.ecapa.tdnn_channels";
const KEY_MFA_CHANNELS: &str = "vokra.ecapa.mfa_channels";
const KEY_ATTENTION_CHANNELS: &str = "vokra.ecapa.attention_channels";
const KEY_RES2NET_SCALE: &str = "vokra.ecapa.res2net_scale";
const KEY_BN_EPS: &str = "vokra.ecapa.bn_eps";
const KEY_STATS_EPS: &str = "vokra.ecapa.stats_eps";
const KEY_FRONTEND: &str = "vokra.ecapa.frontend";
const KEY_PADDING: &str = "vokra.ecapa.padding";
const KEY_LAYOUT: &str = "vokra.ecapa.artifact_layout";
const CONTRACT_KEYS: [&str; 15] = [
    KEY_SAMPLE_RATE,
    KEY_N_MELS,
    KEY_N_FFT,
    KEY_WIN_LENGTH,
    KEY_HOP_LENGTH,
    KEY_EMBED_DIM,
    KEY_TDNN_CHANNELS,
    KEY_MFA_CHANNELS,
    KEY_ATTENTION_CHANNELS,
    KEY_RES2NET_SCALE,
    KEY_BN_EPS,
    KEY_STATS_EPS,
    KEY_FRONTEND,
    KEY_PADDING,
    KEY_LAYOUT,
];

#[derive(Debug)]
pub(crate) struct FoldedBatchNorm {
    scale: Vec<f32>,
    shift: Vec<f32>,
}

impl FoldedBatchNorm {
    pub(crate) fn bind(file: &GgufFile, prefix: &str, channels: usize, eps: f32) -> Result<Self> {
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

    pub(crate) fn apply(&self, values: &mut [f32], frames: usize) {
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
pub(crate) struct Conv1d {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    dilation: usize,
}

impl Conv1d {
    pub(crate) fn bind(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: tensor(
                file,
                &format!("{prefix}.weight"),
                &[output_channels, input_channels, kernel],
            )?,
            bias: tensor(file, &format!("{prefix}.bias"), &[output_channels])?,
            input_channels,
            output_channels,
            kernel,
            dilation,
        })
    }

    pub(crate) fn forward(
        &self,
        input: &[f32],
        frames: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if input.len() != self.input_channels * frames {
            return Err(VokraError::InvalidArgument(format!(
                "ecapa_tdnn: conv input has {} values, expected {} x {frames}",
                input.len(),
                self.input_channels
            )));
        }
        let effective_kernel = (self.kernel - 1) * self.dilation + 1;
        let padding = effective_kernel / 2;
        let padded = reflect_pad(input, self.input_channels, frames, padding)?;
        let expanded_weight = expand_dilated_kernel(
            &self.weight,
            self.output_channels,
            self.input_channels,
            self.kernel,
            self.dilation,
        );
        let mut output = vec![0.0; self.output_channels * frames];
        compute.conv1d_f32(
            &padded,
            self.input_channels,
            frames + 2 * padding,
            &expanded_weight,
            self.output_channels,
            effective_kernel,
            Some(&self.bias),
            1,
            0,
            &mut output,
        )?;
        Ok(output)
    }
}

#[derive(Debug)]
pub(crate) struct TdnnBlock {
    conv: Conv1d,
    norm: FoldedBatchNorm,
}

impl TdnnBlock {
    pub(crate) fn bind(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        dilation: usize,
        bn_eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            conv: Conv1d::bind(
                file,
                &format!("{prefix}.conv.conv"),
                input_channels,
                output_channels,
                kernel,
                dilation,
            )?,
            norm: FoldedBatchNorm::bind(
                file,
                &format!("{prefix}.norm.norm"),
                output_channels,
                bn_eps,
            )?,
        })
    }

    pub(crate) fn forward(
        &self,
        input: &[f32],
        frames: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        let mut output = self.conv.forward(input, frames, compute)?;
        for value in &mut output {
            *value = value.max(0.0);
        }
        self.norm.apply(&mut output, frames);
        Ok(output)
    }
}

#[derive(Debug)]
pub(crate) struct SeRes2NetBlock {
    tdnn1: TdnnBlock,
    res2net: Vec<TdnnBlock>,
    tdnn2: TdnnBlock,
    se_conv1: Conv1d,
    se_conv2: Conv1d,
}

impl SeRes2NetBlock {
    pub(crate) fn bind(
        file: &GgufFile,
        config: &EcapaBackboneConfig,
        block: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self> {
        Self::bind_at(
            file,
            config,
            &config.tensor_name(&format!("blocks.{block}")),
            kernel,
            dilation,
        )
    }

    pub(crate) fn bind_at(
        file: &GgufFile,
        config: &EcapaBackboneConfig,
        prefix: &str,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self> {
        let res2net_channels = config.tdnn_channels / config.res2net_scale;
        let res2net = (0..config.res2net_scale - 1)
            .map(|inner| {
                TdnnBlock::bind(
                    file,
                    &format!("{prefix}.res2net_block.blocks.{inner}"),
                    res2net_channels,
                    res2net_channels,
                    kernel,
                    dilation,
                    config.bn_eps,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            tdnn1: TdnnBlock::bind(
                file,
                &format!("{prefix}.tdnn1"),
                config.tdnn_channels,
                config.tdnn_channels,
                1,
                1,
                config.bn_eps,
            )?,
            res2net,
            tdnn2: TdnnBlock::bind(
                file,
                &format!("{prefix}.tdnn2"),
                config.tdnn_channels,
                config.tdnn_channels,
                1,
                1,
                config.bn_eps,
            )?,
            se_conv1: Conv1d::bind(
                file,
                &format!("{prefix}.se_block.conv1.conv"),
                config.tdnn_channels,
                config.attention_channels,
                1,
                1,
            )?,
            se_conv2: Conv1d::bind(
                file,
                &format!("{prefix}.se_block.conv2.conv"),
                config.attention_channels,
                config.tdnn_channels,
                1,
                1,
            )?,
        })
    }

    pub(crate) fn forward(
        &self,
        input: &[f32],
        frames: usize,
        tdnn_channels: usize,
        res2net_scale: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        let hidden = self.tdnn1.forward(input, frames, compute)?;
        let mut res2 = vec![0.0; tdnn_channels * frames];
        let res2net_channels = tdnn_channels / res2net_scale;
        let chunk_len = res2net_channels * frames;
        res2[..chunk_len].copy_from_slice(&hidden[..chunk_len]);
        let mut previous = Vec::new();
        for chunk in 1..res2net_scale {
            let start = chunk * chunk_len;
            let mut current = hidden[start..start + chunk_len].to_vec();
            if chunk > 1 {
                for (value, residual) in current.iter_mut().zip(&previous) {
                    *value += residual;
                }
            }
            previous = self.res2net[chunk - 1].forward(&current, frames, compute)?;
            res2[start..start + chunk_len].copy_from_slice(&previous);
        }

        let mut output = self.tdnn2.forward(&res2, frames, compute)?;
        let mut squeezed = vec![0.0; tdnn_channels];
        for channel in 0..tdnn_channels {
            squeezed[channel] = output[channel * frames..(channel + 1) * frames]
                .iter()
                .copied()
                .sum::<f32>()
                / frames as f32;
        }
        let mut excitation = self.se_conv1.forward(&squeezed, 1, compute)?;
        for value in &mut excitation {
            *value = value.max(0.0);
        }
        excitation = self.se_conv2.forward(&excitation, 1, compute)?;
        for value in &mut excitation {
            *value = 1.0 / (1.0 + (-*value).exp());
        }
        for (channel, &scale) in excitation.iter().enumerate().take(tdnn_channels) {
            for frame in 0..frames {
                let index = channel * frames + frame;
                output[index] = output[index] * scale + input[index];
            }
        }
        Ok(output)
    }
}

#[derive(Debug)]
pub(crate) struct EcapaBackbone {
    stem: TdnnBlock,
    blocks: Vec<SeRes2NetBlock>,
    mfa: TdnnBlock,
    asp_tdnn: TdnnBlock,
    asp_conv: Conv1d,
    asp_norm: FoldedBatchNorm,
    projection: Conv1d,
    config: EcapaBackboneConfig,
}

impl EcapaBackbone {
    pub(crate) fn bind(file: &GgufFile, config: EcapaBackboneConfig) -> Result<Self> {
        config.validate()?;
        verify_manifest(file, &config)?;
        Ok(Self {
            stem: TdnnBlock::bind(
                file,
                &config.tensor_name("blocks.0"),
                config.input_dim,
                config.tdnn_channels,
                5,
                1,
                config.bn_eps,
            )?,
            blocks: config
                .block_kernels
                .into_iter()
                .zip(config.block_dilations)
                .enumerate()
                .map(|(index, (kernel, dilation))| {
                    SeRes2NetBlock::bind(file, &config, index + 1, kernel, dilation)
                })
                .collect::<Result<Vec<_>>>()?,
            mfa: TdnnBlock::bind(
                file,
                &config.tensor_name("mfa"),
                config.mfa_channels,
                config.mfa_channels,
                1,
                1,
                config.bn_eps,
            )?,
            asp_tdnn: TdnnBlock::bind(
                file,
                &config.tensor_name("asp.tdnn"),
                config.mfa_channels * 3,
                config.attention_channels,
                1,
                1,
                config.bn_eps,
            )?,
            asp_conv: Conv1d::bind(
                file,
                &config.tensor_name("asp.conv.conv"),
                config.attention_channels,
                config.mfa_channels,
                1,
                1,
            )?,
            asp_norm: FoldedBatchNorm::bind(
                file,
                &config.tensor_name("asp_bn.norm"),
                config.mfa_channels * 2,
                config.bn_eps,
            )?,
            projection: Conv1d::bind(
                file,
                &config.tensor_name("fc.conv"),
                config.mfa_channels * 2,
                config.embedding_dim,
                1,
                1,
            )?,
            config,
        })
    }

    pub(crate) fn embed_features(
        &self,
        features: &[f32],
        frames: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if features.len() != frames * self.config.input_dim {
            return Err(VokraError::InvalidArgument(format!(
                "{}: feature buffer has {} values, expected {frames} x {}",
                self.config.diagnostic,
                features.len(),
                self.config.input_dim
            )));
        }
        let max_padding = self.config.max_reflect_padding();
        if frames <= max_padding {
            return Err(VokraError::InvalidArgument(format!(
                "{}: {frames} feature frames are too short for reflect padding {max_padding}",
                self.config.diagnostic
            )));
        }
        let mut channel_major = vec![0.0; self.config.input_dim * frames];
        for frame in 0..frames {
            for channel in 0..self.config.input_dim {
                channel_major[channel * frames + frame] =
                    features[frame * self.config.input_dim + channel];
            }
        }

        let mut hidden = self.stem.forward(&channel_major, frames, compute)?;
        let mut aggregate = Vec::with_capacity(self.config.mfa_channels * frames);
        for block in &self.blocks {
            hidden = block.forward(
                &hidden,
                frames,
                self.config.tdnn_channels,
                self.config.res2net_scale,
                compute,
            )?;
            aggregate.extend_from_slice(&hidden);
        }
        let mfa = self.mfa.forward(&aggregate, frames, compute)?;
        let mut pooled = attentive_statistics_pool(
            &mfa,
            frames,
            self.config.mfa_channels,
            self.config.stats_eps,
            &self.asp_tdnn,
            &self.asp_conv,
            compute,
        )?;
        self.asp_norm.apply(&mut pooled, 1);
        self.projection.forward(&pooled, 1, compute)
    }
}

/// Complete native ECAPA-TDNN inference handle.
#[derive(Debug)]
pub struct EcapaTdnn {
    weights: EcapaBackbone,
    frontend: SpeechbrainFbankAttrs,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl EcapaTdnn {
    /// Strictly binds the exact 200-tensor public SpeechBrain ECAPA manifest.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        require_string(file, chunks::KEY_MODEL_NAME, NAME)?;
        require_string(file, KEY_CATEGORY, CATEGORY)?;
        require_string(file, KEY_UPSTREAM_HF, UPSTREAM_HF)?;
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|value| value.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        let weights = EcapaBackbone::bind(file, EcapaBackboneConfig::speaker())?;
        verify_optional_contract(file)?;
        Ok(Self {
            weights,
            frontend: SpeechbrainFbankAttrs::ecapa_voxceleb(),
            weight_license,
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and strictly binds a GGUF file.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Selects one backend for every learned convolution and attention softmax.
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

    /// Returns the normalized provenance license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Runs the official frontend and complete ECAPA-TDNN speaker network.
    pub fn embed_pcm(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        if sample_rate != self.frontend.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "ecapa_tdnn: expected {} Hz mono PCM, got {sample_rate} Hz; resample offline first",
                self.frontend.sample_rate
            )));
        }
        let (features, frames) = speechbrain_fbank(pcm, &self.frontend)?;
        self.embed_features(&features, frames)
    }

    /// Runs the network on row-major `[frames, 80]` frontend features.
    pub fn embed_features(&self, features: &[f32], frames: usize) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, ECAPA_HOT_OPS)?;
        self.weights.embed_features(features, frames, &compute)
    }

    /// Computes only the official SpeechBrain frontend for parity diagnostics.
    pub fn frontend_features(&self, pcm: &[f32], sample_rate: u32) -> Result<(Vec<f32>, usize)> {
        if sample_rate != self.frontend.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "ecapa_tdnn: expected {} Hz mono PCM, got {sample_rate} Hz",
                self.frontend.sample_rate
            )));
        }
        speechbrain_fbank(pcm, &self.frontend)
    }
}

impl SpeakerEngine for EcapaTdnn {
    fn embed(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        self.embed_pcm(pcm, sample_rate)
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }
}

fn attentive_statistics_pool(
    input: &[f32],
    frames: usize,
    mfa_channels: usize,
    stats_eps: f32,
    attention_tdnn: &TdnnBlock,
    attention_conv: &Conv1d,
    compute: &Compute,
) -> Result<Vec<f32>> {
    if input.len() != mfa_channels * frames || frames == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "ecapa_tdnn: attentive pool expected {} x {frames}, got {} values",
            mfa_channels,
            input.len()
        )));
    }
    let mut context = vec![0.0; mfa_channels * 3 * frames];
    context[..mfa_channels * frames].copy_from_slice(input);
    for channel in 0..mfa_channels {
        let row = &input[channel * frames..(channel + 1) * frames];
        let mean = row.iter().copied().sum::<f32>() / frames as f32;
        let variance = row
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / frames as f32;
        let std = variance.max(stats_eps).sqrt();
        for frame in 0..frames {
            context[(mfa_channels + channel) * frames + frame] = mean;
            context[(mfa_channels * 2 + channel) * frames + frame] = std;
        }
    }

    let mut attention = attention_tdnn.forward(&context, frames, compute)?;
    for value in &mut attention {
        *value = value.tanh();
    }
    let logits = attention_conv.forward(&attention, frames, compute)?;
    let mut weights = vec![0.0; logits.len()];
    compute.softmax_f32(&logits, &mut weights, mfa_channels, frames)?;

    let mut pooled = vec![0.0; mfa_channels * 2];
    for channel in 0..mfa_channels {
        let row = &input[channel * frames..(channel + 1) * frames];
        let probability = &weights[channel * frames..(channel + 1) * frames];
        let mean = row
            .iter()
            .zip(probability)
            .map(|(&value, &weight)| value * weight)
            .sum::<f32>();
        let variance = row
            .iter()
            .zip(probability)
            .map(|(&value, &weight)| {
                let centered = value - mean;
                weight * centered * centered
            })
            .sum::<f32>();
        pooled[channel] = mean;
        pooled[mfa_channels + channel] = variance.max(stats_eps).sqrt();
    }
    Ok(pooled)
}

fn reflect_pad(input: &[f32], channels: usize, frames: usize, padding: usize) -> Result<Vec<f32>> {
    if padding == 0 {
        return Ok(input.to_vec());
    }
    if frames <= padding {
        return Err(VokraError::InvalidArgument(format!(
            "ecapa_tdnn: reflect padding {padding} requires more than {padding} frames, got {frames}"
        )));
    }
    let padded_frames = frames + 2 * padding;
    let mut output = vec![0.0; channels * padded_frames];
    for channel in 0..channels {
        let source = &input[channel * frames..(channel + 1) * frames];
        let destination = &mut output[channel * padded_frames..(channel + 1) * padded_frames];
        for index in 0..padding {
            destination[index] = source[padding - index];
        }
        destination[padding..padding + frames].copy_from_slice(source);
        for index in 0..padding {
            destination[padding + frames + index] = source[frames - 2 - index];
        }
    }
    Ok(output)
}

fn expand_dilated_kernel(
    weight: &[f32],
    output_channels: usize,
    input_channels: usize,
    kernel: usize,
    dilation: usize,
) -> Vec<f32> {
    if dilation == 1 {
        return weight.to_vec();
    }
    let effective_kernel = (kernel - 1) * dilation + 1;
    let mut output = vec![0.0; output_channels * input_channels * effective_kernel];
    for out_channel in 0..output_channels {
        for in_channel in 0..input_channels {
            for tap in 0..kernel {
                output[(out_channel * input_channels + in_channel) * effective_kernel
                    + tap * dilation] =
                    weight[(out_channel * input_channels + in_channel) * kernel + tap];
            }
        }
    }
    output
}

fn expected_manifest(config: &EcapaBackboneConfig) -> Vec<(String, Vec<usize>)> {
    let mut expected = Vec::with_capacity(TENSOR_COUNT);
    push_tdnn(
        &mut expected,
        &config.tensor_name("blocks.0"),
        config.input_dim,
        config.tdnn_channels,
        5,
    );
    let res2net_channels = config.tdnn_channels / config.res2net_scale;
    for block in 1..=3 {
        let prefix = config.tensor_name(&format!("blocks.{block}"));
        push_tdnn(
            &mut expected,
            &format!("{prefix}.tdnn1"),
            config.tdnn_channels,
            config.tdnn_channels,
            1,
        );
        for inner in 0..config.res2net_scale - 1 {
            push_tdnn(
                &mut expected,
                &format!("{prefix}.res2net_block.blocks.{inner}"),
                res2net_channels,
                res2net_channels,
                config.block_kernels[block - 1],
            );
        }
        push_tdnn(
            &mut expected,
            &format!("{prefix}.tdnn2"),
            config.tdnn_channels,
            config.tdnn_channels,
            1,
        );
        push_conv(
            &mut expected,
            &format!("{prefix}.se_block.conv1.conv"),
            config.tdnn_channels,
            config.attention_channels,
            1,
        );
        push_conv(
            &mut expected,
            &format!("{prefix}.se_block.conv2.conv"),
            config.attention_channels,
            config.tdnn_channels,
            1,
        );
    }
    push_tdnn(
        &mut expected,
        &config.tensor_name("mfa"),
        config.mfa_channels,
        config.mfa_channels,
        1,
    );
    push_tdnn(
        &mut expected,
        &config.tensor_name("asp.tdnn"),
        config.mfa_channels * 3,
        config.attention_channels,
        1,
    );
    push_conv(
        &mut expected,
        &config.tensor_name("asp.conv.conv"),
        config.attention_channels,
        config.mfa_channels,
        1,
    );
    push_norm(
        &mut expected,
        &config.tensor_name("asp_bn.norm"),
        config.mfa_channels * 2,
    );
    push_conv(
        &mut expected,
        &config.tensor_name("fc.conv"),
        config.mfa_channels * 2,
        config.embedding_dim,
        1,
    );
    expected
}

fn push_tdnn(
    expected: &mut Vec<(String, Vec<usize>)>,
    prefix: &str,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
) {
    push_conv(
        expected,
        &format!("{prefix}.conv.conv"),
        input_channels,
        output_channels,
        kernel,
    );
    push_norm(expected, &format!("{prefix}.norm.norm"), output_channels);
}

fn push_conv(
    expected: &mut Vec<(String, Vec<usize>)>,
    prefix: &str,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
) {
    expected.push((
        format!("{prefix}.weight"),
        vec![output_channels, input_channels, kernel],
    ));
    expected.push((format!("{prefix}.bias"), vec![output_channels]));
}

fn push_norm(expected: &mut Vec<(String, Vec<usize>)>, prefix: &str, channels: usize) {
    for suffix in ["weight", "bias", "running_mean", "running_var"] {
        expected.push((format!("{prefix}.{suffix}"), vec![channels]));
    }
}

fn verify_manifest(file: &GgufFile, config: &EcapaBackboneConfig) -> Result<()> {
    let backbone_count = file
        .tensors()
        .iter()
        .filter(|tensor| tensor.name.starts_with(config.tensor_prefix))
        .count();
    if backbone_count != TENSOR_COUNT {
        return Err(VokraError::ModelLoad(format!(
            "{}: unsupported ECAPA tensor manifest under prefix `{}`: count={backbone_count}, expected exactly {TENSOR_COUNT}",
            config.diagnostic, config.tensor_prefix
        )));
    }
    let expected = expected_manifest(config);
    debug_assert_eq!(expected.len(), TENSOR_COUNT);
    for (name, dims) in expected {
        check_dims(file, &name, &dims)?;
    }
    Ok(())
}

fn check_dims(file: &GgufFile, name: &str, expected: &[usize]) -> Result<()> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("ecapa_tdnn: missing tensor `{name}`")))?;
    let expected = expected
        .iter()
        .map(|&dimension| dimension as u64)
        .collect::<Vec<_>>();
    if info.dimensions != expected {
        return Err(VokraError::ModelLoad(format!(
            "ecapa_tdnn: tensor `{name}` has dims {:?}, expected {expected:?}",
            info.dimensions
        )));
    }
    Ok(())
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    check_dims(file, name, expected)?;
    file.tensor_f32(name)
        .map_err(|error| VokraError::ModelLoad(format!("ecapa_tdnn: reading `{name}`: {error}")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| VokraError::ModelLoad(format!("ecapa_tdnn: missing string `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "ecapa_tdnn: `{key}` is `{actual}`, expected `{expected}`"
        )));
    }
    Ok(())
}

fn verify_optional_contract(file: &GgufFile) -> Result<()> {
    if !CONTRACT_KEYS.iter().any(|key| file.get(key).is_some()) {
        return Ok(());
    }
    require_string(file, KEY_UPSTREAM_REVISION, UPSTREAM_REVISION)?;
    for (key, expected) in [
        (KEY_SAMPLE_RATE, 16_000),
        (KEY_N_MELS, INPUT_DIM as u32),
        (KEY_N_FFT, 400),
        (KEY_WIN_LENGTH, 400),
        (KEY_HOP_LENGTH, 160),
        (KEY_EMBED_DIM, EMBED_DIM as u32),
        (KEY_TDNN_CHANNELS, TDNN_CHANNELS as u32),
        (KEY_MFA_CHANNELS, MFA_CHANNELS as u32),
        (KEY_ATTENTION_CHANNELS, ATTENTION_CHANNELS as u32),
        (KEY_RES2NET_SCALE, RES2NET_SCALE as u32),
    ] {
        require_u32(file, key, expected)?;
    }
    require_f32(file, KEY_BN_EPS, BN_EPS)?;
    require_f32(file, KEY_STATS_EPS, STATS_EPS)?;
    require_string(file, KEY_FRONTEND, "speechbrain-fbank-v1")?;
    require_string(file, KEY_PADDING, "reflect-same")?;
    require_string(file, KEY_LAYOUT, "speechbrain-ecapa-200-v1")
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(|value| value.as_u64())
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| VokraError::ModelLoad(format!("ecapa_tdnn: missing u32 `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "ecapa_tdnn: `{key}` is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_f32(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    let actual = match file.get(key) {
        Some(GgufMetadataValue::F32(value)) => *value,
        _ => {
            return Err(VokraError::ModelLoad(format!(
                "ecapa_tdnn: missing f32 `{key}`"
            )));
        }
    };
    if actual.to_bits() != expected.to_bits() {
        return Err(VokraError::ModelLoad(format!(
            "ecapa_tdnn: `{key}` is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_is_exactly_the_public_200_tensor_layout() {
        let manifest = expected_manifest(&EcapaBackboneConfig::speaker());
        assert_eq!(manifest.len(), TENSOR_COUNT);
        let mut names = manifest.iter().map(|(name, _)| name).collect::<Vec<_>>();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), TENSOR_COUNT);
        assert!(manifest.iter().any(|(name, dims)| {
            name == "asp.tdnn.conv.conv.weight" && dims == &[128, 9_216, 1]
        }));
    }

    #[test]
    fn prefixed_variant_manifest_uses_configured_axes() {
        let config = EcapaBackboneConfig {
            input_dim: 60,
            tdnn_channels: 1_024,
            res2net_scale: 8,
            mfa_channels: 3_072,
            attention_channels: 128,
            embedding_dim: 256,
            block_kernels: [3, 3, 1],
            block_dilations: [2, 3, 4],
            bn_eps: 1.0e-5,
            stats_eps: 1.0e-12,
            tensor_prefix: "embedding_model.",
            diagnostic: "test_ecapa",
        };
        config.validate().unwrap();
        let manifest = expected_manifest(&config);
        assert_eq!(manifest.len(), TENSOR_COUNT);
        assert!(manifest.iter().any(|(name, dims)| {
            name == "embedding_model.blocks.0.conv.conv.weight" && dims == &[1_024, 60, 5]
        }));
        assert!(manifest.iter().any(|(name, dims)| {
            name == "embedding_model.blocks.3.res2net_block.blocks.0.conv.conv.weight"
                && dims == &[128, 128, 1]
        }));
        assert!(manifest.iter().any(|(name, dims)| {
            name == "embedding_model.fc.conv.weight" && dims == &[256, 6_144, 1]
        }));
    }

    #[test]
    fn res2net_dilations_expand_only_kernel_taps() {
        let expanded = expand_dilated_kernel(&[1.0, 2.0, 3.0], 1, 1, 3, 4);
        assert_eq!(expanded, [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn reflect_padding_matches_torch() {
        let padded = reflect_pad(&[1.0, 2.0, 3.0, 4.0], 1, 4, 2).unwrap();
        assert_eq!(padded, [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn identity_constants_are_pinned() {
        assert_eq!(ARCH, "ecapa_tdnn");
        assert_eq!(EMBED_DIM, 192);
        assert_eq!(UPSTREAM_REVISION.len(), 40);
    }

    #[test]
    fn partial_new_metadata_contract_is_rejected() {
        let mut builder = vokra_core::gguf::GgufBuilder::new();
        builder.add_u32(KEY_SAMPLE_RATE, 16_000);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let error = verify_optional_contract(&file).unwrap_err();
        assert!(error.to_string().contains(KEY_UPSTREAM_REVISION));
    }
}
