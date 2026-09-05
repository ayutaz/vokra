//! Source-shaped CPU AudioVAE kernels for VoxCPM-0.5B.
//!
//! The upstream implementation uses weight-normalised causal convolutions,
//! Snake activations, three dilated residual units per stage and a terminal
//! tanh.  This module keeps those operations typed and independent of the
//! aggregate legacy VAE store.  A production checkpoint must provide every
//! named tensor through the converter binder; no synthesized defaults are
//! provided here.

use vokra_core::backend::BackendKind;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};

/// Learned operations required by the source-shaped AudioVAE path. A caller
/// must preflight this complete set; no operation silently changes backend.
pub const AUDIO_VAE_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::SnakeActivation,
    HotOp::Tanh,
];

/// Exact 0.5B waveform-encoder contract from the pinned `audio_vae.py`.
pub const AUDIO_VAE_SAMPLE_RATE: u32 = 16_000;
/// Encoder channel width before the four downsampling stages.
pub const AUDIO_VAE_ENCODER_DIM: usize = 128;
/// Encoder stride for each causal downsampling stage.
pub const AUDIO_VAE_ENCODER_RATES: [usize; 4] = [2, 5, 8, 8];
/// Number of latent channels emitted by the encoder terminal convolution.
pub const AUDIO_VAE_LATENT_DIM: usize = 64;
/// Waveform samples represented by one latent frame.
pub const AUDIO_VAE_HOP: usize = 640;
/// Prompt PCM chunk size used before AudioVAE encoding.
pub const AUDIO_VAE_PROMPT_CHUNK: usize = 1_280;
/// Decoder stride for each causal upsampling stage.
pub const AUDIO_VAE_DECODER_RATES: [usize; 4] = [8, 8, 5, 2];

/// Weight-normalised causal Conv1d in grouped `[out, in/groups, kernel]`
/// layout. The caller supplies the source `groups` axis explicitly.
#[derive(Debug, Clone)]
pub struct CausalConv1d {
    /// Weight-normalisation scale vector.
    pub weight_g: Vec<f32>,
    /// Weight-normalisation direction tensor in source layout.
    pub weight_v: Vec<f32>,
    /// Per-output-channel bias.
    pub bias: Vec<f32>,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Kernel width.
    pub kernel: usize,
    /// Dilation factor.
    pub dilation: usize,
    /// Convolution stride.
    pub stride: usize,
    /// Left causal padding.
    pub padding: usize,
    /// Group count from the upstream convolution (`groups=1` for dense,
    /// `groups=in_channels` for depthwise layers).
    pub groups: usize,
}

impl CausalConv1d {
    /// Construct a layer after validating the complete upstream tensor shape.
    #[allow(clippy::too_many_arguments)] // A convolution's intrinsic parameter set is explicit here.
    pub fn new(
        weight_g: Vec<f32>,
        weight_v: Vec<f32>,
        bias: Vec<f32>,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        dilation: usize,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Result<Self> {
        if in_channels == 0
            || out_channels == 0
            || kernel == 0
            || dilation == 0
            || stride == 0
            || groups == 0
            || in_channels % groups != 0
            || out_channels % groups != 0
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm audio VAE causal convolution dimensions must be positive".to_owned(),
            ));
        }
        let grouped_inputs = in_channels / groups;
        if weight_v.len() != out_channels * grouped_inputs * kernel
            || weight_g.len() != out_channels
            || bias.len() != out_channels
        {
            return Err(VokraError::InvalidArgument(format!(
                "voxcpm audio VAE causal convolution shape mismatch: v={} g={} bias={} expected v={} g={} bias={}",
                weight_v.len(),
                weight_g.len(),
                bias.len(),
                out_channels * grouped_inputs * kernel,
                out_channels,
                out_channels,
            )));
        }
        Ok(Self {
            weight_g,
            weight_v,
            bias,
            in_channels,
            out_channels,
            kernel,
            dilation,
            stride,
            padding,
            groups,
        })
    }

    fn normalised_weight(&self, output: usize, input: usize, tap: usize) -> f32 {
        let grouped_inputs = self.in_channels / self.groups;
        let base = output * grouped_inputs * self.kernel;
        let mut norm = 0.0f32;
        for index in 0..grouped_inputs * self.kernel {
            let value = self.weight_v[base + index];
            norm += value * value;
        }
        let norm = norm.sqrt().max(f32::MIN_POSITIVE);
        self.weight_g[output] * self.weight_v[base + input * self.kernel + tap] / norm
    }

    /// Apply causal convolution to `[in_channels, time]` samples.
    pub fn forward(&self, input: &[f32], time: usize) -> Result<Vec<f32>> {
        if input.len() != self.in_channels * time || time == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "voxcpm audio VAE causal convolution input {} != {}*{}",
                input.len(),
                self.in_channels,
                time
            )));
        }
        let effective_kernel = (self.kernel - 1) * self.dilation + 1;
        let padded = time + 2 * self.padding;
        let output_time = if padded < effective_kernel {
            0
        } else {
            (padded - effective_kernel) / self.stride + 1
        };
        let mut output = vec![0.0; self.out_channels * output_time];
        for channel in 0..self.out_channels {
            for out_t in 0..output_time {
                let center = out_t * self.stride;
                let mut value = self.bias[channel];
                let outputs_per_group = self.out_channels / self.groups;
                let group = channel / outputs_per_group;
                let first_input = group * (self.in_channels / self.groups);
                for local_input in 0..(self.in_channels / self.groups) {
                    let in_channel = first_input + local_input;
                    for tap in 0..self.kernel {
                        let source = center + tap * self.dilation;
                        let left_pad = 2 * self.padding;
                        if source < left_pad || source - left_pad >= time {
                            continue;
                        }
                        value += self.normalised_weight(channel, local_input, tap)
                            * input[in_channel * time + source - left_pad];
                    }
                }
                output[channel * output_time + out_t] = value;
            }
        }
        Ok(output)
    }

    /// Backend-dispatched convolution. Dilation is lowered to a sparse
    /// effective kernel because the shared Compute conv seam takes a dense
    /// kernel; grouped weights still use the grouped Metal/CPU kernel.
    pub fn forward_with_compute(
        &self,
        input: &[f32],
        time: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if input.len() != self.in_channels * time || time == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm AudioVAE backend convolution input shape mismatch".to_owned(),
            ));
        }
        let effective_kernel = (self.kernel - 1) * self.dilation + 1;
        let padded_len = time + 2 * self.padding;
        let output_time = if padded_len < effective_kernel {
            0
        } else {
            (padded_len - effective_kernel) / self.stride + 1
        };
        let mut padded = vec![0.0; self.in_channels * padded_len];
        for channel in 0..self.in_channels {
            let start = channel * padded_len + 2 * self.padding;
            padded[start..start + time]
                .copy_from_slice(&input[channel * time..(channel + 1) * time]);
        }
        let grouped_inputs = self.in_channels / self.groups;
        let mut weight = vec![0.0; self.out_channels * grouped_inputs * effective_kernel];
        for output in 0..self.out_channels {
            let norm = (0..grouped_inputs * self.kernel)
                .map(|index| {
                    let value = self.weight_v[output * grouped_inputs * self.kernel + index];
                    value * value
                })
                .sum::<f32>()
                .sqrt()
                .max(f32::MIN_POSITIVE);
            for input_channel in 0..grouped_inputs {
                for tap in 0..self.kernel {
                    weight[(output * grouped_inputs + input_channel) * effective_kernel
                        + tap * self.dilation] = self.weight_g[output]
                        * self.weight_v
                            [(output * grouped_inputs + input_channel) * self.kernel + tap]
                        / norm;
                }
            }
        }
        let mut output = vec![0.0; self.out_channels * output_time];
        if self.groups == 1 {
            compute.conv1d_f32(
                &padded,
                self.in_channels,
                padded_len,
                &weight,
                self.out_channels,
                effective_kernel,
                Some(&self.bias),
                self.stride,
                0,
                &mut output,
            )?;
        } else {
            compute.grouped_conv1d_f32(
                &padded,
                self.in_channels,
                padded_len,
                &weight,
                self.out_channels,
                effective_kernel,
                Some(&self.bias),
                self.stride,
                0,
                self.groups,
                &mut output,
            )?;
        }
        Ok(output)
    }
}

/// Snake activation `x + sin²(alpha*x)/(alpha + 1e-9)` used by the 0.5B
/// AudioVAE. The source does not impose a positive-alpha restriction.
#[derive(Debug, Clone)]
pub struct Snake {
    /// Per-channel Snake frequency parameter.
    pub alpha: Vec<f32>,
}

impl Snake {
    /// Apply Snake independently to each channel.
    pub fn forward(&self, values: &mut [f32], time: usize) -> Result<()> {
        if self.alpha.is_empty() || values.len() != self.alpha.len() * time || time == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm audio VAE Snake shape mismatch".to_owned(),
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "voxcpm audio VAE Snake input must be finite".to_owned(),
            ));
        }
        for (channel, &alpha) in self.alpha.iter().enumerate() {
            if !alpha.is_finite() {
                return Err(VokraError::InvalidArgument(
                    "voxcpm audio VAE Snake alpha must be finite".to_owned(),
                ));
            }
            for sample in &mut values[channel * time..(channel + 1) * time] {
                let phase = alpha * *sample;
                let value = *sample + phase.sin() * phase.sin() / (alpha + 1e-9);
                if !value.is_finite() {
                    return Err(VokraError::InvalidArgument(
                        "voxcpm audio VAE Snake output must be finite".to_owned(),
                    ));
                }
                *sample = value;
            }
        }
        Ok(())
    }

    /// Backend-dispatched Snake for the learned encoder path.  The CPU and
    /// Metal arms use the same Compute seam; no unsupported backend is
    /// silently replaced by the scalar implementation.
    pub fn forward_with_compute(
        &self,
        values: &mut [f32],
        time: usize,
        compute: &Compute,
    ) -> Result<()> {
        if self.alpha.is_empty() || values.len() != self.alpha.len() * time || time == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm audio VAE Snake shape mismatch".to_owned(),
            ));
        }
        if self.alpha.iter().any(|x| !x.is_finite())
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm audio VAE Snake inputs must be finite".to_owned(),
            ));
        }
        let input = values.to_vec();
        compute.snake_activation_f32(&input, &self.alpha, self.alpha.len(), time, values)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "voxcpm audio VAE Snake output must be finite".to_owned(),
            ));
        }
        Ok(())
    }
}

/// A residual unit with one source-authenticated dilated convolution.
#[derive(Debug, Clone)]
pub struct ResidualUnit {
    /// Dilated channel-preserving filter convolution.
    pub filter: CausalConv1d,
    /// First channel-wise Snake activation.
    pub activation: Snake,
    /// Snake activation after the filter convolution.
    pub pointwise_activation: Snake,
    /// Pointwise residual projection convolution.
    pub pointwise: CausalConv1d,
}

impl ResidualUnit {
    /// Run activation, convolution and residual addition.
    pub fn forward(&self, input: &[f32], time: usize) -> Result<Vec<f32>> {
        if self.filter.in_channels != self.filter.out_channels {
            return Err(VokraError::InvalidArgument(
                "voxcpm audio VAE residual unit must preserve channels".to_owned(),
            ));
        }
        let mut activated = input.to_vec();
        self.activation.forward(&mut activated, time)?;
        let filtered = self.filter.forward(&activated, time)?;
        if filtered.len() != input.len() {
            return Err(VokraError::InvalidArgument(
                "voxcpm audio VAE residual unit changed time length".to_owned(),
            ));
        }
        let mut filtered = filtered;
        self.pointwise_activation.forward(&mut filtered, time)?;
        let filtered = self.pointwise.forward(&filtered, time)?;
        if filtered.len() != input.len() {
            return Err(VokraError::InvalidArgument(
                "voxcpm audio VAE pointwise residual changed time length".to_owned(),
            ));
        }
        Ok(input.iter().zip(filtered).map(|(a, b)| a + b).collect())
    }

    fn forward_with_compute(
        &self,
        input: &[f32],
        time: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if self.filter.in_channels != self.filter.out_channels {
            return Err(VokraError::InvalidArgument(
                "voxcpm residual unit must preserve channels".to_owned(),
            ));
        }
        let mut activated = input.to_vec();
        self.activation
            .forward_with_compute(&mut activated, time, compute)?;
        let mut filtered = self
            .filter
            .forward_with_compute(&activated, time, compute)?;
        self.pointwise_activation
            .forward_with_compute(&mut filtered, time, compute)?;
        let filtered = self
            .pointwise
            .forward_with_compute(&filtered, time, compute)?;
        if filtered.len() != input.len() {
            return Err(VokraError::InvalidArgument(
                "voxcpm residual backend shape mismatch".to_owned(),
            ));
        }
        Ok(input.iter().zip(filtered).map(|(a, b)| a + b).collect())
    }
}

/// One source encoder stage: three causal residual units, a channel-width
/// Snake, then a strided causal convolution. Channels double after each stage in the
/// 0.5B checkpoint (128, 256, 512, 1024, 2048).
#[derive(Debug, Clone)]
pub struct EncoderStage {
    residuals: [ResidualUnit; 3],
    /// Channel-width Snake activation before the strided convolution.
    pub activation: Snake,
    downsample: CausalConv1d,
}

impl EncoderStage {
    #[allow(dead_code)] // Staged topology is dormant until the complete composite is authorized.
    pub(crate) fn from_source(
        residuals: [ResidualUnit; 3],
        activation: Snake,
        downsample: CausalConv1d,
    ) -> Self {
        Self {
            residuals,
            activation,
            downsample,
        }
    }

    #[allow(dead_code)] // Staged topology validation is dormant until binding is authorized.
    fn validate(&self, channels: usize, rate: usize) -> Result<usize> {
        if self.downsample.in_channels != channels
            || self.downsample.out_channels != channels * 2
            || self.downsample.kernel != rate * 2
            || self.downsample.stride != rate
            || self.downsample.padding != rate.div_ceil(2)
            || self.downsample.groups != 1
            || self.downsample.dilation != 1
        {
            return Err(VokraError::InvalidArgument(format!(
                "voxcpm AudioVAE encoder downsample contract mismatch for rate {rate}"
            )));
        }
        for (index, residual) in self.residuals.iter().enumerate() {
            if residual.filter.in_channels != channels
                || residual.filter.out_channels != channels
                || residual.filter.kernel != 7
                || residual.filter.dilation != 3usize.pow(index as u32)
                || residual.filter.padding != 3 * residual.filter.dilation
                || residual.filter.stride != 1
                || residual.filter.groups != channels
                || residual.activation.alpha.len() != channels
                || residual.pointwise_activation.alpha.len() != channels
                || residual.pointwise.in_channels != channels
                || residual.pointwise.out_channels != channels
                || residual.pointwise.kernel != 1
                || residual.pointwise.dilation != 1
                || residual.pointwise.padding != 0
                || residual.pointwise.stride != 1
                || residual.pointwise.groups != 1
            {
                return Err(VokraError::InvalidArgument(format!(
                    "voxcpm AudioVAE encoder residual {index} contract mismatch"
                )));
            }
        }
        if self.activation.alpha.len() != channels {
            return Err(VokraError::InvalidArgument(
                "voxcpm AudioVAE encoder stage Snake width mismatch".to_owned(),
            ));
        }
        Ok(channels * 2)
    }

    fn forward(&self, input: &[f32], time: usize) -> Result<(Vec<f32>, usize)> {
        let mut values = input.to_vec();
        for residual in &self.residuals {
            values = residual.forward(&values, time)?;
        }
        self.activation.forward(&mut values, time)?;
        let values = self.downsample.forward(&values, time)?;
        let output_time = values.len() / self.downsample.out_channels;
        Ok((values, output_time))
    }

    fn forward_with_compute(
        &self,
        input: &[f32],
        time: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        let mut values = input.to_vec();
        for residual in &self.residuals {
            values = residual.forward_with_compute(&values, time, compute)?;
        }
        self.activation
            .forward_with_compute(&mut values, time, compute)?;
        let values = self
            .downsample
            .forward_with_compute(&values, time, compute)?;
        let output_time = values.len() / self.downsample.out_channels;
        Ok((values, output_time))
    }
}

/// Source-exact waveform → continuous latent AudioVAE encoder for the 0.5B
/// release.  It consumes mono 16-kHz channel-major PCM and emits
/// `[latent_dim, frames]`; it does not normalize, resample, or synthesize
/// missing prompt samples.
#[derive(Debug, Clone)]
pub struct AudioVaeEncoder {
    stem: CausalConv1d,
    stages: Vec<EncoderStage>,
    /// `fc_mu` (kernel=3, padding=1) is the encoded latent path. The
    /// companion `fc_logvar` tensors are not evaluated by source `encode`,
    /// but a future strict composite binder must authenticate their complete
    /// names/shapes before treating the encoder bundle as production-ready.
    terminal: CausalConv1d,
}

/// Apply the source preprocessing rule: right-zero-pad to an integral VAE
/// hop before any convolution.  Keeping this separate makes the frame
/// contract testable without constructing the full learned checkpoint.
fn pad_audio_vae_pcm(pcm: &[f32], samples: usize) -> Result<(Vec<f32>, usize)> {
    if samples == 0 || pcm.len() != samples || pcm.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "voxcpm AudioVAE encoder requires finite mono PCM".to_owned(),
        ));
    }
    let padded_samples = audio_vae_frame_count(samples)?
        .checked_mul(AUDIO_VAE_HOP)
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "voxcpm AudioVAE encoder PCM length overflows hop padding".to_owned(),
            )
        })?;
    let mut padded = vec![0.0f32; padded_samples];
    padded[..samples].copy_from_slice(pcm);
    Ok((padded, padded_samples))
}

/// Prepare source prompt audio before the VAE encoder. VoxCPM pads to one
/// complete two-frame patch (`patch_size * chunk_size = 1280`) before VAE
/// encoding; the final complete patch is removed by the prompt packer.
#[allow(dead_code)] // Prompt packing awaits the complete authenticated VoxCPM route.
pub(crate) fn pad_audio_vae_prompt_pcm(pcm: &[f32]) -> Result<Vec<f32>> {
    if pcm.is_empty() || pcm.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "voxcpm prompt PCM must be finite and non-empty".to_owned(),
        ));
    }
    let padded_samples = pcm
        .len()
        .checked_add(AUDIO_VAE_PROMPT_CHUNK - 1)
        .map(|value| value / AUDIO_VAE_PROMPT_CHUNK)
        .and_then(|chunks| chunks.checked_mul(AUDIO_VAE_PROMPT_CHUNK))
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "voxcpm prompt PCM length overflows chunk padding".to_owned(),
            )
        })?;
    let mut padded = vec![0.0; padded_samples];
    padded[..pcm.len()].copy_from_slice(pcm);
    Ok(padded)
}

fn audio_vae_frame_count(samples: usize) -> Result<usize> {
    if samples == 0 {
        return Err(VokraError::InvalidArgument(
            "voxcpm AudioVAE frame count requires nonzero samples".to_owned(),
        ));
    }
    samples
        .checked_add(AUDIO_VAE_HOP - 1)
        .map(|value| value / AUDIO_VAE_HOP)
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "voxcpm AudioVAE encoder PCM length overflows hop padding".to_owned(),
            )
        })
}

impl AudioVaeEncoder {
    /// Attach source-shaped encoder layers after strict tensor binding.  The
    /// dimensions and rates are fixed by the pinned 0.5B config; no
    /// zero/default layer can be used as a production substitute.
    #[allow(dead_code)] // Staged topology constructor awaits complete composite authorization.
    pub(crate) fn from_source(
        stem: CausalConv1d,
        stages: Vec<EncoderStage>,
        terminal: CausalConv1d,
    ) -> Result<Self> {
        if stem.in_channels != 1
            || stem.out_channels != AUDIO_VAE_ENCODER_DIM
            || stem.kernel != 7
            || stem.dilation != 1
            || stem.stride != 1
            || stem.padding != 3
            || stem.groups != 1
            || stages.len() != AUDIO_VAE_ENCODER_RATES.len()
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm AudioVAE encoder stem/rate contract mismatch".to_owned(),
            ));
        }
        let mut channels = AUDIO_VAE_ENCODER_DIM;
        for (stage, &rate) in stages.iter().zip(AUDIO_VAE_ENCODER_RATES.iter()) {
            channels = stage.validate(channels, rate)?;
        }
        if terminal.in_channels != channels
            || terminal.out_channels != AUDIO_VAE_LATENT_DIM
            || terminal.kernel != 3
            || terminal.dilation != 1
            || terminal.stride != 1
            || terminal.padding != 1
            || terminal.groups != 1
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm AudioVAE encoder terminal contract mismatch".to_owned(),
            ));
        }
        Ok(Self {
            stem,
            stages,
            terminal,
        })
    }

    /// Attach a VAST-staged encoder bundle after the converter has resolved
    /// the official state-dict names.  The converter must pass every
    /// `[out,in/groups,kernel]`, weight-norm vector and bias explicitly; this
    /// constructor never discovers tensors by prefix or fills missing layers.
    /// It is crate-private until the complete composite manifest records the
    /// release's exact encoder name mapping.
    #[allow(dead_code)] // Staged topology constructor awaits complete composite authorization.
    pub(crate) fn from_staged_parts(
        stem: CausalConv1d,
        stages: Vec<EncoderStage>,
        terminal: CausalConv1d,
    ) -> Result<Self> {
        Self::from_source(stem, stages, terminal)
    }

    /// Encode mono channel-major 16-kHz PCM with the scalar reference path.
    pub fn encode(&self, pcm: &[f32], samples: usize) -> Result<Vec<f32>> {
        let (padded_pcm, padded_samples) = pad_audio_vae_pcm(pcm, samples)?;
        let mut values = self.stem.forward(&padded_pcm, padded_samples)?;
        let mut time = values.len() / self.stem.out_channels;
        for stage in &self.stages {
            (values, time) = stage.forward(&values, time)?;
        }
        self.terminal.forward(&values, time)
    }

    /// Encode through one explicitly selected backend.  Every convolution
    /// and Snake activation is dispatched through Compute; no CPU fallback is
    /// available when a GPU backend is selected.
    pub fn encode_with_compute(
        &self,
        pcm: &[f32],
        samples: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        let (padded_pcm, padded_samples) = pad_audio_vae_pcm(pcm, samples)?;
        let mut values = self
            .stem
            .forward_with_compute(&padded_pcm, padded_samples, compute)?;
        let mut time = values.len() / self.stem.out_channels;
        for stage in &self.stages {
            (values, time) = stage.forward_with_compute(&values, time, compute)?;
        }
        self.terminal.forward_with_compute(&values, time, compute)
    }

    /// Encode through the explicitly selected backend kind.
    pub fn encode_with_backend(
        &self,
        pcm: &[f32],
        samples: usize,
        backend: BackendKind,
    ) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(backend, AUDIO_VAE_HOT_OPS)?;
        self.encode_with_compute(pcm, samples, &compute)
    }
}

/// One decoder stage: Snake, causal upsample, then three residual units.
#[derive(Debug, Clone)]
pub struct DecoderStage {
    /// Channel-wise Snake activation before upsampling.
    pub activation: Snake,
    /// Causal transposed convolution that upsamples the stage.
    pub upsample: CausalConvTranspose1d,
    /// Three channel-preserving residual units.
    pub residuals: [ResidualUnit; 3],
}

impl DecoderStage {
    #[allow(dead_code)] // Staged topology is dormant until the complete composite is authorized.
    pub(crate) fn from_source(
        activation: Snake,
        upsample: CausalConvTranspose1d,
        residuals: [ResidualUnit; 3],
    ) -> Self {
        Self {
            activation,
            upsample,
            residuals,
        }
    }

    #[allow(dead_code)] // Staged topology validation is dormant until binding is authorized.
    fn validate(&self, channels: usize, rate: usize) -> Result<usize> {
        let next_channels = channels / 2;
        if channels == 0
            || channels % 2 != 0
            || self.activation.alpha.len() != channels
            || self.upsample.in_channels != channels
            || self.upsample.out_channels != next_channels
            || self.upsample.kernel != 2 * rate
            || self.upsample.stride != rate
            || self.upsample.groups != 1
        {
            return Err(VokraError::InvalidArgument(format!(
                "voxcpm AudioVAE decoder stage contract mismatch for rate {rate}"
            )));
        }
        for (index, residual) in self.residuals.iter().enumerate() {
            if residual.filter.in_channels != next_channels
                || residual.filter.out_channels != next_channels
                || residual.filter.kernel != 7
                || residual.filter.dilation != 3usize.pow(index as u32)
                || residual.filter.padding != 3 * residual.filter.dilation
                || residual.filter.stride != 1
                || residual.filter.groups != next_channels
                || residual.activation.alpha.len() != next_channels
                || residual.pointwise_activation.alpha.len() != next_channels
                || residual.pointwise.in_channels != next_channels
                || residual.pointwise.out_channels != next_channels
                || residual.pointwise.kernel != 1
                || residual.pointwise.dilation != 1
                || residual.pointwise.padding != 0
                || residual.pointwise.stride != 1
                || residual.pointwise.groups != 1
            {
                return Err(VokraError::InvalidArgument(format!(
                    "voxcpm AudioVAE decoder residual {index} contract mismatch"
                )));
            }
        }
        Ok(next_channels)
    }

    /// Run the source stage in channel-first layout.
    pub fn forward(&self, input: &[f32], time: usize) -> Result<(Vec<f32>, usize)> {
        let mut activated = input.to_vec();
        self.activation.forward(&mut activated, time)?;
        let (mut values, next_time) = self.upsample.forward(&activated, time)?;
        for residual in &self.residuals {
            values = residual.forward(&values, next_time)?;
        }
        Ok((values, next_time))
    }

    fn forward_with_compute(
        &self,
        input: &[f32],
        time: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        let mut activated = input.to_vec();
        self.activation
            .forward_with_compute(&mut activated, time, compute)?;
        let (mut values, next_time) = self
            .upsample
            .forward_with_compute(&activated, time, compute)?;
        for residual in &self.residuals {
            values = residual.forward_with_compute(&values, next_time, compute)?;
        }
        Ok((values, next_time))
    }
}

/// Causal transposed Conv1d used for decoder upsampling.
#[derive(Debug, Clone)]
pub struct CausalConvTranspose1d {
    /// Weight-normalisation scale vector.
    pub weight_g: Vec<f32>,
    /// Weight-normalisation direction tensor in source layout.
    pub weight_v: Vec<f32>,
    /// Per-output-channel bias.
    pub bias: Vec<f32>,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Kernel width.
    pub kernel: usize,
    /// Upsampling stride.
    pub stride: usize,
    /// Number of channel groups.
    pub groups: usize,
}

impl CausalConvTranspose1d {
    /// Construct a transposed convolution with explicit source tensor shapes.
    #[allow(clippy::too_many_arguments)] // A transposed convolution's intrinsic parameter set is explicit here.
    pub fn new(
        weight_g: Vec<f32>,
        weight_v: Vec<f32>,
        bias: Vec<f32>,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        if in_channels == 0
            || out_channels == 0
            || kernel == 0
            || stride == 0
            || groups == 0
            || in_channels % groups != 0
            || out_channels % groups != 0
            || weight_g.len() != in_channels
            || weight_v.len() != in_channels * (out_channels / groups) * kernel
            || bias.len() != out_channels
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm audio VAE causal transpose convolution shape mismatch".to_owned(),
            ));
        }
        Ok(Self {
            weight_g,
            weight_v,
            bias,
            in_channels,
            out_channels,
            kernel,
            stride,
            groups,
        })
    }

    /// Apply causal transposed convolution, returning channel-first samples.
    pub fn forward(&self, input: &[f32], time: usize) -> Result<(Vec<f32>, usize)> {
        if input.len() != self.in_channels * time || time == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm audio VAE transpose input shape mismatch".to_owned(),
            ));
        }
        // The pinned source uses ConvTranspose1d(padding=0,
        // output_padding=0), followed by a right crop.  Do not fold the
        // crop into the raw PyTorch output-length formula.
        let output_padding = 0usize;
        let raw_time = (time - 1)
            .checked_mul(self.stride)
            .and_then(|value| value.checked_add(self.kernel))
            .and_then(|value| value.checked_add(output_padding))
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "voxcpm audio VAE transpose output shape underflow/overflow".to_owned(),
                )
            })?;
        let crop = 2 * self.stride.div_ceil(2) - (self.stride % 2);
        let output_time = raw_time.checked_sub(crop).ok_or_else(|| {
            VokraError::InvalidArgument(
                "voxcpm audio VAE transpose crop exceeds raw output".to_owned(),
            )
        })?;
        let mut output = vec![0.0; self.out_channels * raw_time];
        let outputs_per_group = self.out_channels / self.groups;
        for channel in 0..self.in_channels {
            let base = channel * outputs_per_group * self.kernel;
            let norm = self.weight_v[base..base + outputs_per_group * self.kernel]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
                .max(f32::MIN_POSITIVE);
            for input_t in 0..time {
                for out_channel in 0..self.out_channels {
                    let group = channel / (self.in_channels / self.groups);
                    if out_channel / outputs_per_group != group {
                        continue;
                    }
                    let local_output = out_channel % outputs_per_group;
                    for tap in 0..self.kernel {
                        let output_t = input_t * self.stride + tap;
                        if output_t >= raw_time {
                            continue;
                        }
                        let weight = self.weight_g[channel]
                            * self.weight_v[base + local_output * self.kernel + tap]
                            / norm;
                        output[out_channel * raw_time + output_t] +=
                            input[channel * time + input_t] * weight;
                    }
                }
            }
        }
        for channel in 0..self.out_channels {
            for sample in &mut output[channel * raw_time..(channel + 1) * raw_time] {
                *sample += self.bias[channel];
            }
        }
        let mut cropped = vec![0.0; self.out_channels * output_time];
        for channel in 0..self.out_channels {
            let source = &output[channel * raw_time..(channel + 1) * raw_time];
            let destination = &mut cropped[channel * output_time..(channel + 1) * output_time];
            destination.copy_from_slice(&source[..output_time]);
        }
        Ok((cropped, output_time))
    }

    /// Backend-dispatched transpose convolution. Compute has no transpose
    /// kernel, so each learned input/output projection is lowered to GEMM;
    /// the causal scatter/crop remains deterministic layout glue.
    pub fn forward_with_compute(
        &self,
        input: &[f32],
        time: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        if input.len() != self.in_channels * time || time == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm AudioVAE backend transpose input shape mismatch".to_owned(),
            ));
        }
        let output_padding = 0usize;
        let raw_time = (time - 1)
            .checked_mul(self.stride)
            .and_then(|value| value.checked_add(self.kernel))
            .and_then(|value| value.checked_add(output_padding))
            .ok_or_else(|| {
                VokraError::InvalidArgument("voxcpm transpose shape overflow".to_owned())
            })?;
        let crop = 2 * self.stride.div_ceil(2) - (self.stride % 2);
        let output_time = raw_time.checked_sub(crop).ok_or_else(|| {
            VokraError::InvalidArgument("voxcpm transpose crop exceeds raw output".to_owned())
        })?;
        let outputs_per_group = self.out_channels / self.groups;
        let mut output = vec![0.0; self.out_channels * raw_time];
        let mut matrix = vec![0.0; self.in_channels * self.out_channels * self.kernel];
        for input_channel in 0..self.in_channels {
            let group = input_channel / (self.in_channels / self.groups);
            let norm = self.weight_v[input_channel * outputs_per_group * self.kernel
                ..(input_channel + 1) * outputs_per_group * self.kernel]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
                .max(f32::MIN_POSITIVE);
            for output_channel in 0..self.out_channels {
                if output_channel / outputs_per_group != group {
                    continue;
                }
                let local = output_channel % outputs_per_group;
                for tap in 0..self.kernel {
                    matrix[input_channel * self.out_channels * self.kernel
                        + output_channel * self.kernel
                        + tap] = self.weight_g[input_channel]
                        * self.weight_v[input_channel * outputs_per_group * self.kernel
                            + local * self.kernel
                            + tap]
                        / norm;
                }
            }
        }
        for input_t in 0..time {
            let mut projected = vec![0.0; self.out_channels * self.kernel];
            compute.gemm_f32(
                1,
                self.out_channels * self.kernel,
                self.in_channels,
                &(0..self.in_channels)
                    .map(|channel| input[channel * time + input_t])
                    .collect::<Vec<_>>(),
                &matrix,
                None,
                &mut projected,
            )?;
            for output_channel in 0..self.out_channels {
                for tap in 0..self.kernel {
                    let position = input_t * self.stride + tap;
                    if position < raw_time {
                        output[output_channel * raw_time + position] +=
                            projected[output_channel * self.kernel + tap];
                    }
                }
            }
        }
        for channel in 0..self.out_channels {
            for value in &mut output[channel * raw_time..(channel + 1) * raw_time] {
                *value += self.bias[channel];
            }
        }
        let mut cropped = vec![0.0; self.out_channels * output_time];
        for channel in 0..self.out_channels {
            cropped[channel * output_time..(channel + 1) * output_time]
                .copy_from_slice(&output[channel * raw_time..channel * raw_time + output_time]);
        }
        Ok((cropped, output_time))
    }
}

/// Complete source-shaped decoder including terminal Snake, convolution and tanh.
#[derive(Debug, Clone)]
pub struct AudioVaeDecoder {
    /// Initial latent-dimension to decoder-width causal k7 stem.
    pub stem: CausalConv1d,
    /// Decoder stages that progressively upsample the latent sequence.
    pub stages: Vec<DecoderStage>,
    /// Terminal waveform projection convolution.
    pub terminal: CausalConv1d,
    /// Terminal channel-wise Snake activation.
    pub terminal_activation: Snake,
}

impl AudioVaeDecoder {
    /// Attach the exact 0.5B decoder topology after every learned tensor has
    /// been resolved by the composite binder. This validates source names'
    /// resulting axes but does not authenticate an arbitrary GGUF by itself.
    #[allow(dead_code)] // Staged topology constructor awaits complete composite authorization.
    pub(crate) fn from_source(
        stem: CausalConv1d,
        stages: Vec<DecoderStage>,
        terminal: CausalConv1d,
        terminal_activation: Snake,
    ) -> Result<Self> {
        if stem.in_channels != AUDIO_VAE_LATENT_DIM
            || stem.out_channels != 1536
            || stem.kernel != 7
            || stem.dilation != 1
            || stem.stride != 1
            || stem.padding != 3
            || stem.groups != 1
            || stages.len() != AUDIO_VAE_DECODER_RATES.len()
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm AudioVAE decoder stem/rate contract mismatch".to_owned(),
            ));
        }
        let mut channels = stem.out_channels;
        for (stage, &rate) in stages.iter().zip(AUDIO_VAE_DECODER_RATES.iter()) {
            channels = stage.validate(channels, rate)?;
        }
        if channels != 96
            || terminal_activation.alpha.len() != channels
            || terminal.in_channels != channels
            || terminal.out_channels != 1
            || terminal.kernel != 7
            || terminal.dilation != 1
            || terminal.stride != 1
            || terminal.padding != 3
            || terminal.groups != 1
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm AudioVAE decoder terminal contract mismatch".to_owned(),
            ));
        }
        Ok(Self {
            stem,
            stages,
            terminal,
            terminal_activation,
        })
    }

    #[allow(dead_code)] // Staged topology constructor awaits complete composite authorization.
    pub(crate) fn from_staged_parts(
        stem: CausalConv1d,
        stages: Vec<DecoderStage>,
        terminal: CausalConv1d,
        terminal_activation: Snake,
    ) -> Result<Self> {
        Self::from_source(stem, stages, terminal, terminal_activation)
    }

    /// Decode `[latent_dim,time]` into mono PCM. All weights are required.
    pub fn decode(&self, latents: &[f32], time: usize) -> Result<Vec<f32>> {
        if latents.is_empty() || time == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm audio VAE decoder input is empty".to_owned(),
            ));
        }
        let mut values = self.stem.forward(latents, time)?;
        let mut current_time = time;
        for stage in &self.stages {
            (values, current_time) = stage.forward(&values, current_time)?;
        }
        self.terminal_activation
            .forward(&mut values, current_time)?;
        values = self.terminal.forward(&values, current_time)?;
        for value in &mut values {
            *value = value.tanh();
        }
        Ok(values)
    }

    /// Decode through one preflighted selected backend. Host-side activation
    /// and causal layout glue are deterministic; all learned convolutions are
    /// dispatched through `compute`.
    pub fn decode_with_compute(
        &self,
        latents: &[f32],
        time: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if latents.is_empty() || time == 0 || latents.len() != self.stem.in_channels * time {
            return Err(VokraError::InvalidArgument(
                "voxcpm AudioVAE backend decoder input shape mismatch".to_owned(),
            ));
        }
        let mut values = self.stem.forward_with_compute(latents, time, compute)?;
        let stem_effective = (self.stem.kernel - 1) * self.stem.dilation + 1;
        let mut current_time =
            (time + 2 * self.stem.padding - stem_effective) / self.stem.stride + 1;
        for stage in &self.stages {
            (values, current_time) = stage.forward_with_compute(&values, current_time, compute)?;
        }
        self.terminal_activation
            .forward_with_compute(&mut values, current_time, compute)?;
        values = self
            .terminal
            .forward_with_compute(&values, current_time, compute)?;
        let pre_tanh = values.clone();
        compute.tanh_f32(&pre_tanh, &mut values)?;
        Ok(values)
    }

    /// Build the selected backend once and decode without per-op fallback.
    pub fn decode_with_backend(
        &self,
        latents: &[f32],
        time: usize,
        backend: BackendKind,
    ) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(backend, AUDIO_VAE_HOT_OPS)?;
        self.decode_with_compute(latents, time, &compute)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(channels: usize, stride: usize) -> CausalConv1d {
        CausalConv1d::new(
            vec![1.0; channels],
            vec![1.0; channels * channels],
            vec![0.0; channels],
            channels,
            channels,
            1,
            1,
            stride,
            1,
            1,
        )
        .unwrap()
    }

    #[test]
    fn causal_stride_length_and_shape_are_explicit() {
        let layer = conv(1, 2);
        assert_eq!(
            layer.forward(&[1.0, 2.0, 3.0], 3).unwrap(),
            vec![0.0, 1.0, 3.0]
        );
        assert!(layer.forward(&[1.0], 3).is_err());
    }

    #[test]
    fn snake_matches_source_eps_and_accepts_finite_zero_or_negative_alpha() {
        let snake = Snake { alpha: vec![1.0] };
        let mut values = vec![0.5, -0.5];
        snake.forward(&mut values, 2).unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
        let mut zero = [0.0];
        Snake { alpha: vec![0.0] }.forward(&mut zero, 1).unwrap();
        assert_eq!(zero, [0.0]);
        let mut negative = [0.25];
        Snake { alpha: vec![-1.0] }
            .forward(&mut negative, 1)
            .unwrap();
        assert!(negative[0].is_finite());
        assert!(
            Snake {
                alpha: vec![f32::NAN]
            }
            .forward(&mut [0.0], 1)
            .is_err()
        );
    }

    #[test]
    fn encoder_contract_pins_16khz_rates_and_rejects_partial_layers() {
        assert_eq!(AUDIO_VAE_SAMPLE_RATE, 16_000);
        assert_eq!(AUDIO_VAE_ENCODER_DIM, 128);
        assert_eq!(AUDIO_VAE_ENCODER_RATES, [2, 5, 8, 8]);
        assert_eq!(AUDIO_VAE_LATENT_DIM, 64);
        assert_eq!(AUDIO_VAE_HOP, 640);
        assert_eq!(AUDIO_VAE_DECODER_RATES, [8, 8, 5, 2]);
        assert_eq!(
            AUDIO_VAE_DECODER_RATES.iter().product::<usize>(),
            AUDIO_VAE_HOP
        );
        let stem =
            CausalConv1d::new(vec![1.0], vec![1.0; 7], vec![0.0], 1, 1, 7, 1, 1, 3, 1).unwrap();
        let terminal =
            CausalConv1d::new(vec![1.0], vec![1.0; 3], vec![0.0], 1, 1, 3, 1, 1, 1, 1).unwrap();
        assert!(AudioVaeEncoder::from_source(stem, Vec::new(), terminal).is_err());
    }

    #[test]
    fn encoder_preprocess_right_pads_to_hop_and_reports_frames() {
        let pcm = [1.0f32; 640];
        let (padded, samples) = pad_audio_vae_pcm(&pcm, pcm.len()).unwrap();
        assert_eq!(samples, 640);
        assert_eq!(padded.len(), 640);
        assert_eq!(audio_vae_frame_count(samples).unwrap(), 1);

        let pcm = [1.0f32; 641];
        let (padded, samples) = pad_audio_vae_pcm(&pcm, pcm.len()).unwrap();
        assert_eq!(samples, 1280);
        assert_eq!(padded.len(), 1280);
        assert_eq!(&padded[..641], &pcm);
        assert!(padded[641..].iter().all(|value| *value == 0.0));
        assert_eq!(audio_vae_frame_count(641).unwrap(), 2);
    }

    #[test]
    fn prompt_preprocess_pads_to_two_frame_chunk() {
        let exact = vec![1.0f32; AUDIO_VAE_PROMPT_CHUNK];
        let padded = pad_audio_vae_prompt_pcm(&exact).unwrap();
        assert_eq!(padded.len(), AUDIO_VAE_PROMPT_CHUNK);

        let short = vec![2.0f32; AUDIO_VAE_PROMPT_CHUNK - 1];
        let padded = pad_audio_vae_prompt_pcm(&short).unwrap();
        assert_eq!(padded.len(), AUDIO_VAE_PROMPT_CHUNK);
        assert_eq!(&padded[..short.len()], &short);
        assert_eq!(padded[short.len()], 0.0);

        let longer = vec![3.0f32; AUDIO_VAE_PROMPT_CHUNK + 1];
        assert_eq!(
            pad_audio_vae_prompt_pcm(&longer).unwrap().len(),
            AUDIO_VAE_PROMPT_CHUNK * 2
        );
    }

    #[test]
    fn snake_backend_path_is_explicitly_compute_dispatched() {
        let snake = Snake { alpha: vec![1.0] };
        let mut values = vec![0.25, -0.5];
        snake
            .forward_with_compute(&mut values, 2, &Compute::cpu())
            .unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn transpose_stride_and_terminal_tanh_are_bounded() {
        let transpose = CausalConvTranspose1d::new(
            vec![1.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.0],
            1,
            1,
            4,
            2,
            1,
        )
        .unwrap();
        let (values, time) = transpose.forward(&[1.0, 2.0], 2).unwrap();
        assert_eq!(time, 4);
        // Kernel width four at stride two overlaps the two input frames in
        // the middle; the normalized contributions therefore sum to 1.5.
        assert_eq!(values, vec![0.5, 0.5, 1.5, 1.5]);
        let terminal = (10.0f32).tanh();
        assert!(terminal.is_finite());
        assert!((-1.0..=1.0).contains(&terminal));
    }

    #[test]
    fn decoder_source_rates_produce_exact_audio_hop() {
        let mut values = vec![1.0f32];
        let mut time = 1usize;
        for &rate in &AUDIO_VAE_DECODER_RATES {
            let previous_time = time;
            let layer = CausalConvTranspose1d::new(
                vec![1.0],
                vec![1.0; rate * 2],
                vec![0.0],
                1,
                1,
                rate * 2,
                rate,
                1,
            )
            .unwrap();
            (values, time) = layer.forward(&values, time).unwrap();
            assert_eq!(time, previous_time * rate);
        }
        assert_eq!(time, AUDIO_VAE_HOP);
    }
}
