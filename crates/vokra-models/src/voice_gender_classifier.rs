//! Native JaesungHuh voice-gender classifier (CPU / Metal).
//!
//! This is deliberately a distinct architecture from the SpeechBrain ECAPA
//! speaker encoder.  The official `model.py` uses a 512-point torchaudio
//! frontend, a `conv1` / `layer1..3` SE-Res2Net stack, a 1536-wide MFA
//! projection, 4608-wide attentive statistics pooling, and `fc6` / `fc7`
//! binary classification layers.  It must never be accepted by the
//! `ecapa_tdnn` binder (the historical public artifact had exactly that
//! incorrect arch stamp).
//!
//! Upstream source: `JaesungHuh/voice-gender-classifier` at immutable commit
//! `49bcbecfd929ba5a043bde645fdff1a375eb79c7`.  The upstream repository and
//! checkpoint are MIT according to their primary source declarations.  The
//! runtime consumes only safetensors-derived GGUF and routes learned Conv1d,
//! GEMV and softmax operations through [`Compute`].

use std::path::Path;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::ir::graph::{Normalization, PadMode, StftAttrs, Window, WindowSymmetry};
use vokra_core::{CompliancePolicy, LicenseClass, Result, VokraError, check_weight_license};
use vokra_ops::stft;

use crate::compute::{Compute, HotOp};
use crate::ecapa_tdnn::{EcapaBackboneConfig, FoldedBatchNorm};

/// GGUF architecture tag for the dedicated classifier.
pub const ARCH: &str = "voice_gender_classifier";
/// Human-readable model name stamped by the converter.
pub const NAME: &str = "voice-gender-classifier";
/// Model-zoo category.
pub const CATEGORY: &str = "classification";
/// Canonical upstream Hugging Face repository.
pub const UPSTREAM_HF: &str = "JaesungHuh/voice-gender-classifier";
/// Pinned upstream source revision.
pub const UPSTREAM_REVISION: &str = "49bcbecfd929ba5a043bde645fdff1a375eb79c7";
/// Pinned upstream checkpoint revision.
pub const UPSTREAM_HF_REVISION: &str = "db1222153bd60337e900be22add7af180452adc0";
/// Required PCM sampling rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// Number of mel features.
pub const N_MELS: usize = 80;
/// Embedding width before the classifier head.
pub const EMBED_DIM: usize = 192;
/// Number of classifier outputs.
pub const CLASS_COUNT: usize = 2;
/// Official output-label order.
pub const CLASS_LABELS: [&str; CLASS_COUNT] = ["male", "female"];
/// Learned operation inventory used for backend preflight.
pub const HOT_OPS: &[HotOp] = &[HotOp::Conv1d, HotOp::Gemv, HotOp::Softmax];

const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const F_MIN: f32 = 20.0;
const F_MAX: f32 = 7_600.0;
const TDNN_CHANNELS: usize = 1_024;
const MFA_CHANNELS: usize = 1_536;
const ATTENTION_CHANNELS: usize = 256;
const RES2NET_SCALE: usize = 8;
const BN_EPS: f32 = 1.0e-5;
const VAR_EPS: f32 = 1.0e-4;
const TENSOR_COUNT: usize = 202;

const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.voice_gender.upstream_revision";
const KEY_UPSTREAM_HF_REVISION: &str = "vokra.voice_gender.upstream_hf_revision";
const KEY_SAMPLE_RATE: &str = "vokra.voice_gender.sample_rate";
const KEY_N_MELS: &str = "vokra.voice_gender.n_mels";
const KEY_N_FFT: &str = "vokra.voice_gender.n_fft";
const KEY_WIN_LENGTH: &str = "vokra.voice_gender.win_length";
const KEY_HOP_LENGTH: &str = "vokra.voice_gender.hop_length";
const KEY_F_MIN: &str = "vokra.voice_gender.f_min";
const KEY_F_MAX: &str = "vokra.voice_gender.f_max";
const KEY_TDNN_CHANNELS: &str = "vokra.voice_gender.tdnn_channels";
const KEY_MFA_CHANNELS: &str = "vokra.voice_gender.mfa_channels";
const KEY_ATTENTION_CHANNELS: &str = "vokra.voice_gender.attention_channels";
const KEY_EMBED_DIM: &str = "vokra.voice_gender.embed_dim";
const KEY_CLASS_COUNT: &str = "vokra.voice_gender.class_count";
const KEY_LABELS: &str = "vokra.voice_gender.labels";
const KEY_FRONTEND: &str = "vokra.voice_gender.frontend";
const KEY_ARTIFACT_LAYOUT: &str = "vokra.voice_gender.artifact_layout";

#[derive(Debug)]
struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
    name: &'static str,
}

impl Linear {
    fn bind(file: &GgufFile, name: &'static str, input: usize, output: usize) -> Result<Self> {
        Ok(Self {
            weight: tensor(file, &format!("{name}.weight"), &[output, input])?,
            bias: tensor(file, &format!("{name}.bias"), &[output])?,
            input,
            output,
            name,
        })
    }

    fn forward(&self, input: &[f32], compute: &Compute) -> Result<Vec<f32>> {
        if input.len() != self.input {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: {} input width {}, expected {}",
                self.name,
                input.len(),
                self.input
            )));
        }
        let mut output = vec![0.0; self.output];
        compute.gemv_f32(
            self.output,
            self.input,
            &self.weight,
            input,
            Some(&self.bias),
            &mut output,
        )?;
        Ok(output)
    }
}

#[derive(Debug)]
struct Bottle2neck {
    conv1: ZeroPadBlock,
    convs: Vec<ZeroPadBlock>,
    conv3: ZeroPadBlock,
    se1: ZeroPadConv1d,
    se2: ZeroPadConv1d,
}

/// Applies the official Bottle2neck branch order: transform spx[0], feed
/// each later transformed branch into the next residual sum, then append the
/// final spx branch unchanged.
fn res2net_branches<F>(
    hidden: &[f32],
    width: usize,
    frames: usize,
    mut transform: F,
) -> Result<Vec<f32>>
where
    F: FnMut(usize, &[f32]) -> Result<Vec<f32>>,
{
    let branch_count = RES2NET_SCALE - 1;
    let chunk_len = width * frames;
    if hidden.len() != (branch_count + 1) * chunk_len {
        return Err(VokraError::InvalidArgument(format!(
            "{ARCH}: Bottle2neck branch input has {} values, expected {}",
            hidden.len(),
            (branch_count + 1) * chunk_len
        )));
    }
    let mut output = vec![0.0; hidden.len()];
    let mut previous = transform(0, &hidden[..chunk_len])?;
    if previous.len() != chunk_len {
        return Err(VokraError::InvalidArgument(format!(
            "{ARCH}: Bottle2neck branch output has {}, expected {chunk_len}",
            previous.len()
        )));
    }
    output[..chunk_len].copy_from_slice(&previous);
    for index in 1..branch_count {
        let start = index * chunk_len;
        let mut current = hidden[start..start + chunk_len].to_vec();
        for (value, residual) in current.iter_mut().zip(&previous) {
            *value += residual;
        }
        previous = transform(index, &current)?;
        if previous.len() != chunk_len {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: Bottle2neck branch output has {}, expected {chunk_len}",
                previous.len()
            )));
        }
        output[start..start + chunk_len].copy_from_slice(&previous);
    }
    let final_start = branch_count * chunk_len;
    output[final_start..].copy_from_slice(&hidden[final_start..]);
    Ok(output)
}

impl Bottle2neck {
    fn bind(file: &GgufFile, config: &EcapaBackboneConfig, layer: usize) -> Result<Self> {
        let prefix = format!("layer{layer}");
        let width = config.tdnn_channels / config.res2net_scale;
        let conv1 = tdnn_parts(
            file,
            &format!("{prefix}.conv1"),
            &format!("{prefix}.bn1"),
            config.tdnn_channels,
            config.tdnn_channels,
            1,
            1,
            config.bn_eps,
        )?;
        let convs = (0..config.res2net_scale - 1)
            .map(|index| {
                tdnn_parts(
                    file,
                    &format!("{prefix}.convs.{index}"),
                    &format!("{prefix}.bns.{index}"),
                    width,
                    width,
                    3,
                    [2, 3, 4][layer - 1],
                    config.bn_eps,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let conv3 = tdnn_parts(
            file,
            &format!("{prefix}.conv3"),
            &format!("{prefix}.bn3"),
            config.tdnn_channels,
            config.tdnn_channels,
            1,
            1,
            config.bn_eps,
        )?;
        Ok(Self {
            conv1,
            convs,
            conv3,
            se1: ZeroPadConv1d::bind(
                file,
                &format!("{prefix}.se.se.1"),
                config.tdnn_channels,
                128,
                1,
                1,
            )?,
            se2: ZeroPadConv1d::bind(
                file,
                &format!("{prefix}.se.se.3"),
                128,
                config.tdnn_channels,
                1,
                1,
            )?,
        })
    }

    fn forward(&self, input: &[f32], frames: usize, compute: &Compute) -> Result<Vec<f32>> {
        let channels = self.conv1_channels();
        let width = channels / self.convs.len().saturating_add(1);
        let hidden = self.conv1.forward(input, frames, compute)?;
        let out = res2net_branches(&hidden, width, frames, |index, branch| {
            self.convs[index].forward(branch, frames, compute)
        })?;
        let mut output = self.conv3.forward(&out, frames, compute)?;
        let mut squeezed = vec![0.0; channels];
        for channel in 0..channels {
            squeezed[channel] = output[channel * frames..(channel + 1) * frames]
                .iter()
                .copied()
                .sum::<f32>()
                / frames as f32;
        }
        let mut excitation = self.se1.forward(&squeezed, 1, compute)?;
        for value in &mut excitation {
            *value = value.max(0.0);
        }
        let excitation = self.se2.forward(&excitation, 1, compute)?;
        for (channel, &scale) in excitation.iter().enumerate() {
            let scale = 1.0 / (1.0 + (-scale).exp());
            for frame in 0..frames {
                output[channel * frames + frame] =
                    output[channel * frames + frame] * scale + input[channel * frames + frame];
            }
        }
        Ok(output)
    }

    fn conv1_channels(&self) -> usize {
        // The stem and all Bottle2neck branches in this model are fixed at C=1024.
        TDNN_CHANNELS
    }
}

fn tdnn_parts(
    file: &GgufFile,
    conv_prefix: &str,
    norm_prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    dilation: usize,
    eps: f32,
) -> Result<ZeroPadBlock> {
    let conv = ZeroPadConv1d::bind(file, conv_prefix, input, output, kernel, dilation)?;
    let norm = FoldedBatchNorm::bind(file, norm_prefix, output, eps)?;
    Ok(ZeroPadBlock { conv, norm })
}

#[derive(Debug)]
struct ZeroPadBlock {
    conv: ZeroPadConv1d,
    norm: FoldedBatchNorm,
}

impl ZeroPadBlock {
    fn forward(&self, input: &[f32], frames: usize, compute: &Compute) -> Result<Vec<f32>> {
        let mut output = self.conv.forward(input, frames, compute)?;
        for value in &mut output {
            *value = value.max(0.0);
        }
        self.norm.apply(&mut output, frames);
        Ok(output)
    }
}

/// Conv1d matching ordinary PyTorch `nn.Conv1d(..., padding=...)`: samples
/// outside the time axis are zero, never reflected. The shared ECAPA helper
/// intentionally implements SpeechBrain's reflect-same contract and is not
/// suitable for this upstream model.
#[derive(Debug)]
struct ZeroPadConv1d {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    dilation: usize,
}

impl ZeroPadConv1d {
    fn bind(
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

    fn forward(&self, input: &[f32], frames: usize, compute: &Compute) -> Result<Vec<f32>> {
        if frames == 0 || input.len() != self.input_channels * frames {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: zero-pad conv input has {} values, expected {} x {frames}",
                input.len(),
                self.input_channels
            )));
        }
        let effective_kernel = (self.kernel - 1) * self.dilation + 1;
        let padding = effective_kernel / 2;
        let expanded_weight = expand_dilated_kernel(
            &self.weight,
            self.output_channels,
            self.input_channels,
            self.kernel,
            self.dilation,
        );
        let mut output = vec![0.0; self.output_channels * frames];
        compute.conv1d_f32(
            input,
            self.input_channels,
            frames,
            &expanded_weight,
            self.output_channels,
            effective_kernel,
            Some(&self.bias),
            1,
            padding,
            &mut output,
        )?;
        Ok(output)
    }
}

fn expand_dilated_kernel(
    weight: &[f32],
    output_channels: usize,
    input_channels: usize,
    kernel: usize,
    dilation: usize,
) -> Vec<f32> {
    let effective_kernel = (kernel - 1) * dilation + 1;
    let mut expanded = vec![0.0; output_channels * input_channels * effective_kernel];
    for output in 0..output_channels {
        for input in 0..input_channels {
            for tap in 0..kernel {
                expanded[(output * input_channels + input) * effective_kernel + tap * dilation] =
                    weight[(output * input_channels + input) * kernel + tap];
            }
        }
    }
    expanded
}

/// Dedicated native voice-gender classifier runtime.
#[derive(Debug)]
pub struct VoiceGenderClassifier {
    stem: ZeroPadBlock,
    layers: Vec<Bottle2neck>,
    layer4: ZeroPadConv1d,
    attention1: ZeroPadConv1d,
    attention_norm: FoldedBatchNorm,
    attention2: ZeroPadConv1d,
    bn5: FoldedBatchNorm,
    fc6: Linear,
    bn6: FoldedBatchNorm,
    fc7: Linear,
    frontend: StftAttrs,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl VoiceGenderClassifier {
    /// Binds a strictly stamped GGUF classifier.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        require_string(file, chunks::KEY_MODEL_NAME, NAME)?;
        require_string(file, "vokra.model.category", CATEGORY)?;
        require_string(file, KEY_UPSTREAM_HF, UPSTREAM_HF)?;
        let license = check_weight_license(file, &CompliancePolicy::strict())?;
        require_string(
            file,
            vokra_core::gguf::chunks::KEY_PROVENANCE_LICENSE,
            "mit",
        )?;
        if license.class != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: MIT weight must resolve to permissive, got {}",
                license.class.as_str()
            )));
        }
        verify_metadata(file)?;
        if file.tensors().len() != TENSOR_COUNT {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: expected exactly {TENSOR_COUNT} tensors, got {}",
                file.tensors().len()
            )));
        }
        let config = EcapaBackboneConfig {
            input_dim: N_MELS,
            tdnn_channels: TDNN_CHANNELS,
            res2net_scale: RES2NET_SCALE,
            mfa_channels: TDNN_CHANNELS * 3,
            attention_channels: ATTENTION_CHANNELS,
            embedding_dim: EMBED_DIM,
            block_kernels: [3, 3, 3],
            block_dilations: [2, 3, 4],
            bn_eps: BN_EPS,
            stats_eps: VAR_EPS,
            tensor_prefix: "",
            diagnostic: ARCH,
        };
        config.validate()?;
        let stem = tdnn_parts(file, "conv1", "bn1", N_MELS, TDNN_CHANNELS, 5, 1, BN_EPS)?;
        let layers = (1..=3)
            .map(|layer| Bottle2neck::bind(file, &config, layer))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            stem,
            layers,
            layer4: ZeroPadConv1d::bind(file, "layer4", TDNN_CHANNELS * 3, MFA_CHANNELS, 1, 1)?,
            attention1: ZeroPadConv1d::bind(
                file,
                "attention.0",
                MFA_CHANNELS * 3,
                ATTENTION_CHANNELS,
                1,
                1,
            )?,
            attention_norm: FoldedBatchNorm::bind(file, "attention.2", ATTENTION_CHANNELS, BN_EPS)?,
            attention2: ZeroPadConv1d::bind(
                file,
                "attention.4",
                ATTENTION_CHANNELS,
                MFA_CHANNELS,
                1,
                1,
            )?,
            bn5: FoldedBatchNorm::bind(file, "bn5", MFA_CHANNELS * 2, BN_EPS)?,
            fc6: Linear::bind(file, "fc6", MFA_CHANNELS * 2, EMBED_DIM)?,
            bn6: FoldedBatchNorm::bind(file, "bn6", EMBED_DIM, BN_EPS)?,
            fc7: Linear::bind(file, "fc7", EMBED_DIM, CLASS_COUNT)?,
            frontend: frontend_attrs(),
            weight_license: license.class,
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and binds a classifier from a GGUF path.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Selects the backend used by learned operations.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the selected compute backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the stamped weight license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Computes normalized frontend features and their frame count.
    pub fn frontend_features(&self, pcm: &[f32], sample_rate: u32) -> Result<(Vec<f32>, usize)> {
        if sample_rate != SAMPLE_RATE {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: expected {SAMPLE_RATE} Hz mono PCM, got {sample_rate} Hz"
            )));
        }
        gender_fbank(pcm, &self.frontend)
    }

    /// Computes classifier logits from PCM input.
    pub fn logits_pcm(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let (features, frames) = self.frontend_features(pcm, sample_rate)?;
        self.logits_features(&features, frames)
    }

    /// Computes classifier logits from row-major frontend features.
    pub fn logits_features(&self, features: &[f32], frames: usize) -> Result<Vec<f32>> {
        let (_, logits) = self.embedding_and_logits_features(features, frames)?;
        Ok(logits)
    }

    /// Returns the post-normalization embedding used by the classifier head.
    pub fn embedding_features(&self, features: &[f32], frames: usize) -> Result<Vec<f32>> {
        let (embedding, _) = self.embedding_and_logits_features(features, frames)?;
        Ok(embedding)
    }

    fn embedding_and_logits_features(
        &self,
        features: &[f32],
        frames: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if frames <= 1 || features.len() != frames * N_MELS {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: expected at least 2 frames and row-major {frames} x {N_MELS} features"
            )));
        }
        let compute = Compute::for_backend(self.backend, HOT_OPS)?;
        let stem = self
            .stem
            .forward(&row_to_channels(features, frames), frames, &compute)?;
        let x1 = self.layers[0].forward(&stem, frames, &compute)?;
        let x0_plus_x1 = stem
            .iter()
            .zip(&x1)
            .map(|(&left, &right)| left + right)
            .collect::<Vec<_>>();
        let x2 = self.layers[1].forward(&x0_plus_x1, frames, &compute)?;
        let x0_plus_x1_plus_x2 = x0_plus_x1
            .iter()
            .zip(&x2)
            .map(|(&left, &right)| left + right)
            .collect::<Vec<_>>();
        let x3 = self.layers[2].forward(&x0_plus_x1_plus_x2, frames, &compute)?;
        let mut aggregate = Vec::with_capacity(TDNN_CHANNELS * 3 * frames);
        aggregate.extend_from_slice(&x1);
        aggregate.extend_from_slice(&x2);
        aggregate.extend_from_slice(&x3);
        let mut x = self.layer4.forward(&aggregate, frames, &compute)?;
        for value in &mut x {
            *value = value.max(0.0);
        }
        let pooled = self.attentive_pool(&x, frames, &compute)?;
        let mut pooled = pooled;
        self.bn5.apply(&mut pooled, 1);
        let mut embedding = self.fc6.forward(&pooled, &compute)?;
        self.bn6.apply(&mut embedding, 1);
        for value in &mut embedding {
            *value = value.max(0.0);
        }
        let logits = self.fc7.forward(&embedding, &compute)?;
        Ok((embedding, logits))
    }

    /// Classifies mono PCM into the official male/female probability order.
    pub fn classify_pcm(&self, pcm: &[f32], sample_rate: u32) -> Result<GenderPrediction> {
        let logits = self.logits_pcm(pcm, sample_rate)?;
        GenderPrediction::from_logits(&logits)
    }

    fn attentive_pool(&self, input: &[f32], frames: usize, compute: &Compute) -> Result<Vec<f32>> {
        let mut context = vec![0.0; MFA_CHANNELS * 3 * frames];
        context[..MFA_CHANNELS * frames].copy_from_slice(input);
        for channel in 0..MFA_CHANNELS {
            let row = &input[channel * frames..(channel + 1) * frames];
            let mean = row.iter().copied().sum::<f32>() / frames as f32;
            let variance =
                row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / (frames - 1) as f32;
            let std = variance.max(VAR_EPS).sqrt();
            for frame in 0..frames {
                context[(MFA_CHANNELS + channel) * frames + frame] = mean;
                context[(MFA_CHANNELS * 2 + channel) * frames + frame] = std;
            }
        }
        let mut attention = self.attention1.forward(&context, frames, compute)?;
        for value in &mut attention {
            *value = value.max(0.0);
        }
        self.attention_norm.apply(&mut attention, frames);
        for value in &mut attention {
            *value = value.tanh();
        }
        let logits = self.attention2.forward(&attention, frames, compute)?;
        let mut weights = vec![0.0; logits.len()];
        compute.softmax_f32(&logits, &mut weights, MFA_CHANNELS, frames)?;
        let mut pooled = vec![0.0; MFA_CHANNELS * 2];
        for channel in 0..MFA_CHANNELS {
            let values = &input[channel * frames..(channel + 1) * frames];
            let probs = &weights[channel * frames..(channel + 1) * frames];
            let mean = values
                .iter()
                .zip(probs)
                .map(|(&value, &prob)| value * prob)
                .sum::<f32>();
            // Match the official `sum((x**2) * w) - mu**2` expression
            // literally; algebraically equivalent weighted deviations can
            // round differently on CPU/Metal.
            let variance = values
                .iter()
                .zip(probs)
                .map(|(&value, &prob)| value * value * prob)
                .sum::<f32>()
                - mean * mean;
            pooled[channel] = mean;
            pooled[MFA_CHANNELS + channel] = variance.max(VAR_EPS).sqrt();
        }
        Ok(pooled)
    }
}

/// Binary voice-gender prediction in official class order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenderPrediction {
    /// Selected label, with ties resolved to `male`.
    pub label: &'static str,
    /// Softmax probabilities ordered as `[male, female]`.
    pub probabilities: [f32; CLASS_COUNT],
}

impl GenderPrediction {
    fn from_logits(logits: &[f32]) -> Result<Self> {
        if logits.len() != CLASS_COUNT || logits.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: classifier returned invalid {}-class logits",
                CLASS_COUNT
            )));
        }
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities = [0.0; CLASS_COUNT];
        let mut total = 0.0;
        for (index, value) in logits.iter().enumerate() {
            probabilities[index] = (*value - maximum).exp();
            total += probabilities[index];
        }
        for value in &mut probabilities {
            *value /= total;
        }
        let index = usize::from(probabilities[1] > probabilities[0]);
        Ok(Self {
            label: CLASS_LABELS[index],
            probabilities,
        })
    }
}

fn frontend_attrs() -> StftAttrs {
    let mut attrs = StftAttrs::new(N_FFT, HOP_LENGTH);
    attrs.win_length = WIN_LENGTH;
    attrs.window = Window::Hamming;
    attrs.window_symmetry = WindowSymmetry::Periodic;
    attrs.center = true;
    attrs.pad_mode = PadMode::Reflect;
    attrs.normalization = Normalization::Backward;
    attrs.real_input = true;
    attrs
}

fn gender_fbank(pcm: &[f32], frontend: &StftAttrs) -> Result<(Vec<f32>, usize)> {
    if pcm.len() < 2 {
        return Err(VokraError::InvalidArgument(format!(
            "{ARCH}: at least 2 PCM samples required"
        )));
    }
    let mut emphasized = vec![0.0; pcm.len()];
    emphasized[0] = pcm[0] - 0.97 * pcm[1];
    for index in 1..pcm.len() {
        emphasized[index] = pcm[index] - 0.97 * pcm[index - 1];
    }
    let spectrum = stft(&emphasized, frontend)?;
    let filters = mel_filters();
    let power = spectrum.power();
    let mut features = vec![0.0; spectrum.frames * N_MELS];
    for frame in 0..spectrum.frames {
        for mel in 0..N_MELS {
            let energy = power[frame * spectrum.bins..(frame + 1) * spectrum.bins]
                .iter()
                .zip(&filters[mel * spectrum.bins..(mel + 1) * spectrum.bins])
                .map(|(&value, &weight)| value * weight)
                .sum::<f32>();
            features[frame * N_MELS + mel] = (energy + 1.0e-6).ln();
        }
    }
    for mel in 0..N_MELS {
        let mean = (0..spectrum.frames)
            .map(|frame| features[frame * N_MELS + mel])
            .sum::<f32>()
            / spectrum.frames as f32;
        for frame in 0..spectrum.frames {
            features[frame * N_MELS + mel] -= mean;
        }
    }
    Ok((features, spectrum.frames))
}

fn mel_filters() -> Vec<f32> {
    let bins = N_FFT / 2 + 1;
    let hz_to_mel = |hz: f32| 2_595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2_595.0) - 1.0);
    let low = hz_to_mel(F_MIN);
    let high = hz_to_mel(F_MAX);
    let points = (0..N_MELS + 2)
        .map(|index| mel_to_hz(low + (high - low) * index as f32 / (N_MELS + 1) as f32))
        .collect::<Vec<_>>();
    let frequencies = (0..bins)
        .map(|index| index as f32 * SAMPLE_RATE as f32 / N_FFT as f32)
        .collect::<Vec<_>>();
    let mut output = vec![0.0; N_MELS * bins];
    for mel in 0..N_MELS {
        let left = points[mel];
        let center = points[mel + 1];
        let right = points[mel + 2];
        for (bin, &frequency) in frequencies.iter().enumerate() {
            let lower = (frequency - left) / (center - left);
            let upper = (right - frequency) / (right - center);
            output[mel * bins + bin] = lower.min(upper).max(0.0).min(1.0);
        }
    }
    output
}

fn row_to_channels(input: &[f32], frames: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for frame in 0..frames {
        for channel in 0..N_MELS {
            output[channel * frames + frame] = input[frame * N_MELS + channel];
        }
    }
    output
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("{ARCH}: missing tensor `{name}`")))?;
    let dimensions = expected
        .iter()
        .map(|&value| value as u64)
        .collect::<Vec<_>>();
    if info.dimensions != dimensions {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: `{name}` dims {:?}, expected {dimensions:?}",
            info.dimensions
        )));
    }
    file.tensor_f32(name)
        .map_err(|error| VokraError::ModelLoad(format!("{ARCH}: reading `{name}`: {error}")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| VokraError::ModelLoad(format!("{ARCH}: missing `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: `{key}` is `{actual}`, expected `{expected}`"
        )));
    }
    Ok(())
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(|value| value.as_u64())
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| VokraError::ModelLoad(format!("{ARCH}: missing `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: `{key}` is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_f32(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    let actual = match file.get(key) {
        Some(GgufMetadataValue::F32(value)) => *value,
        _ => return Err(VokraError::ModelLoad(format!("{ARCH}: missing `{key}`"))),
    };
    if actual.to_bits() != expected.to_bits() {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: `{key}` is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn verify_metadata(file: &GgufFile) -> Result<()> {
    require_string(file, KEY_UPSTREAM_REVISION, UPSTREAM_REVISION)?;
    require_string(file, KEY_UPSTREAM_HF_REVISION, UPSTREAM_HF_REVISION)?;
    for (key, value) in [
        (KEY_SAMPLE_RATE, SAMPLE_RATE as u32),
        (KEY_N_MELS, N_MELS as u32),
        (KEY_N_FFT, N_FFT as u32),
        (KEY_WIN_LENGTH, WIN_LENGTH as u32),
        (KEY_HOP_LENGTH, HOP_LENGTH as u32),
        (KEY_TDNN_CHANNELS, TDNN_CHANNELS as u32),
        (KEY_MFA_CHANNELS, MFA_CHANNELS as u32),
        (KEY_ATTENTION_CHANNELS, ATTENTION_CHANNELS as u32),
        (KEY_EMBED_DIM, EMBED_DIM as u32),
        (KEY_CLASS_COUNT, CLASS_COUNT as u32),
    ] {
        require_u32(file, key, value)?;
    }
    require_f32(file, KEY_F_MIN, F_MIN)?;
    require_f32(file, KEY_F_MAX, F_MAX)?;
    require_string(file, KEY_LABELS, "male,female")?;
    require_string(file, KEY_FRONTEND, "torchaudio-mel-v1")?;
    require_string(file, KEY_ARTIFACT_LAYOUT, "voice-gender-classifier-202-v1")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_pad_conv_does_not_reflect_boundary_samples() {
        let conv = ZeroPadConv1d {
            weight: vec![1.0, 1.0, 1.0],
            bias: vec![0.0],
            input_channels: 1,
            output_channels: 1,
            kernel: 3,
            dilation: 1,
        };
        let compute = Compute::for_backend(BackendKind::Cpu, &[HotOp::Conv1d]).unwrap();
        let output = conv.forward(&[1.0, 2.0], 2, &compute).unwrap();
        // PyTorch Conv1d padding=1 pads [1, 2] as [0, 1, 2, 0]. Reflect
        // padding would use [2, 1, 2, 1], proving this path is not reflect.
        assert_eq!(output, vec![3.0, 3.0]);
    }

    #[test]
    fn bottle2neck_transforms_first_branch_and_leaves_last_raw() {
        let hidden = (1..=8).map(|value| value as f32).collect::<Vec<_>>();
        let output = res2net_branches(&hidden, 1, 1, |index, branch| {
            Ok(vec![branch[0] + 10.0 * (index + 1) as f32])
        })
        .unwrap();
        assert_eq!(
            output,
            vec![11.0, 33.0, 66.0, 110.0, 165.0, 231.0, 308.0, 8.0]
        );
    }
}
