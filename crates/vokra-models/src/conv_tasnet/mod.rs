//! Strict native Asteroid Conv-TasNet Libri1Mix enhancement inference.
//!
//! The official checkpoint is a 512-filter waveform encoder, a 24-block
//! dilated temporal convolutional masker and a learned transposed-convolution
//! decoder. CPU and Metal use the same backend-dispatched Conv1D, grouped
//! Conv1D and LayerNorm operations. Layout changes, residual additions,
//! scalar PReLU/ReLU and zero padding are host control/pointwise glue.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};

/// GGUF architecture tag.
pub const ARCH: &str = "conv_tasnet";
/// Canonical Vokra model identifier.
pub const NAME: &str = "conv-tasnet-libri1mix";
/// Single-stream speech-enhancement task category.
pub const CATEGORY: &str = "enhancement";
/// Pinned official upstream checkpoint repository.
pub const UPSTREAM_HF: &str = "JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k";
/// Pinned upstream revision used by prep and parity.
pub const UPSTREAM_REVISION: &str = "bb8a876bc157b5cf3c405994accb798c49146016";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_MODEL_ID: &str = "vokra.provenance.model_id";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_N_FILTERS: &str = "vokra.conv_tasnet.n_filters";
const KEY_N_KERNEL: &str = "vokra.conv_tasnet.n_kernel";
const KEY_STRIDE: &str = "vokra.conv_tasnet.stride";
const KEY_N_BLOCKS: &str = "vokra.conv_tasnet.n_blocks";
const KEY_N_REPEATS: &str = "vokra.conv_tasnet.n_repeats";
const KEY_BN_CHAN: &str = "vokra.conv_tasnet.bn_chan";
const KEY_HID_CHAN: &str = "vokra.conv_tasnet.hid_chan";
const KEY_SKIP_CHAN: &str = "vokra.conv_tasnet.skip_chan";
const KEY_CONV_KERNEL_SIZE: &str = "vokra.conv_tasnet.conv_kernel_size";
const KEY_SAMPLE_RATE: &str = "vokra.conv_tasnet.sample_rate";
const KEY_N_SRC: &str = "vokra.conv_tasnet.n_src";
const KEY_CAUSAL: &str = "vokra.conv_tasnet.causal";

const FILTERS: usize = 512;
const ENCODER_KERNEL: usize = 32;
const ENCODER_STRIDE: usize = 16;
const BLOCKS: usize = 8;
const REPEATS: usize = 3;
const BOTTLENECK: usize = 128;
const HIDDEN: usize = 512;
const SKIP: usize = 128;
const TCN_KERNEL: usize = 3;
const SAMPLE_RATE: u32 = 16_000;
const SOURCES: usize = 1;
const TENSOR_COUNT: usize = 345;
const GLN_EPS: f32 = 1e-8;

/// Complete backend-op requirement for Conv-TasNet CPU/Metal inference.
pub const CONV_TASNET_HOT_OPS: &[HotOp] = &[HotOp::Conv1d, HotOp::GroupedConv1d, HotOp::LayerNorm];

/// Exact official Asteroid Libri1Mix topology stamped in GGUF metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTasnetConfig {
    /// Learned encoder filter count.
    pub n_filters: u32,
    /// Encoder/decoder kernel width in samples.
    pub n_kernel: u32,
    /// Encoder/decoder stride in samples.
    pub stride: u32,
    /// Dilated TCN blocks per repeat.
    pub n_blocks: u32,
    /// Number of complete TCN repeats.
    pub n_repeats: u32,
    /// Bottleneck channel count.
    pub bn_chan: u32,
    /// Hidden channel count in each depthwise block.
    pub hid_chan: u32,
    /// Skip-connection channel count.
    pub skip_chan: u32,
    /// Undilated depthwise convolution kernel width.
    pub conv_kernel_size: u32,
    /// Required waveform sample rate.
    pub sample_rate: u32,
    /// Number of output waveform streams.
    pub n_src: u32,
    /// Causal flag; the pinned official checkpoint is non-causal (`0`).
    pub causal: u32,
}

impl Default for ConvTasnetConfig {
    fn default() -> Self {
        Self::asteroid_libri1mix_default()
    }
}

impl ConvTasnetConfig {
    /// Official `JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k` axes.
    #[must_use]
    pub const fn asteroid_libri1mix_default() -> Self {
        Self {
            n_filters: FILTERS as u32,
            n_kernel: ENCODER_KERNEL as u32,
            stride: ENCODER_STRIDE as u32,
            n_blocks: BLOCKS as u32,
            n_repeats: REPEATS as u32,
            bn_chan: BOTTLENECK as u32,
            hid_chan: HIDDEN as u32,
            skip_chan: SKIP as u32,
            conv_kernel_size: TCN_KERNEL as u32,
            sample_rate: SAMPLE_RATE,
            n_src: SOURCES as u32,
            causal: 0,
        }
    }

    fn from_gguf(file: &GgufFile) -> Result<Self> {
        let config = Self {
            n_filters: required_u32(file, KEY_N_FILTERS)?,
            n_kernel: required_u32(file, KEY_N_KERNEL)?,
            stride: required_u32(file, KEY_STRIDE)?,
            n_blocks: required_u32(file, KEY_N_BLOCKS)?,
            n_repeats: required_u32(file, KEY_N_REPEATS)?,
            bn_chan: required_u32(file, KEY_BN_CHAN)?,
            hid_chan: required_u32(file, KEY_HID_CHAN)?,
            skip_chan: required_u32(file, KEY_SKIP_CHAN)?,
            conv_kernel_size: required_u32(file, KEY_CONV_KERNEL_SIZE)?,
            sample_rate: required_u32(file, KEY_SAMPLE_RATE)?,
            n_src: required_u32(file, KEY_N_SRC)?,
            causal: required_u32(file, KEY_CAUSAL)?,
        };
        let expected = Self::asteroid_libri1mix_default();
        if config != expected {
            return Err(VokraError::ModelLoad(format!(
                "conv_tasnet: topology {config:?} does not match pinned official checkpoint {expected:?}; reconvert with the current converter"
            )));
        }
        Ok(config)
    }
}

#[derive(Debug)]
struct AffineNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl AffineNorm {
    fn bind(file: &GgufFile, prefix: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            gamma: tensor(file, &format!("{prefix}.gamma"), &[channels])?,
            beta: tensor(file, &format!("{prefix}.beta"), &[channels])?,
        })
    }
}

#[derive(Debug)]
struct Conv1d {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
}

impl Conv1d {
    fn bind(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            weight: tensor(
                file,
                &format!("{prefix}.weight"),
                &[output_channels, input_channels, kernel],
            )?,
            bias: bias
                .then(|| tensor(file, &format!("{prefix}.bias"), &[output_channels]))
                .transpose()?,
            input_channels,
            output_channels,
            kernel,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        input_length: usize,
        stride: usize,
        padding: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        if input_length + 2 * padding < self.kernel || stride == 0 {
            return Err(VokraError::InvalidArgument(
                "conv_tasnet: invalid Conv1D extent".to_owned(),
            ));
        }
        let output_length = (input_length + 2 * padding - self.kernel) / stride + 1;
        let mut output = vec![0.0; self.output_channels * output_length];
        compute.conv1d_f32(
            input,
            self.input_channels,
            input_length,
            &self.weight,
            self.output_channels,
            self.kernel,
            self.bias.as_deref(),
            stride,
            padding,
            &mut output,
        )?;
        Ok((output, output_length))
    }
}

#[derive(Debug)]
struct TcnBlock {
    input: Conv1d,
    prelu_input: f32,
    norm_input: AffineNorm,
    depth_weight: Vec<f32>,
    depth_bias: Vec<f32>,
    prelu_depth: f32,
    norm_depth: AffineNorm,
    residual: Conv1d,
    skip: Conv1d,
    dilation: usize,
}

impl TcnBlock {
    fn bind(file: &GgufFile, index: usize) -> Result<Self> {
        let prefix = format!("masker.TCN.{index}");
        Ok(Self {
            input: Conv1d::bind(
                file,
                &format!("{prefix}.shared_block.0"),
                BOTTLENECK,
                HIDDEN,
                1,
                true,
            )?,
            prelu_input: tensor(file, &format!("{prefix}.shared_block.1.weight"), &[1])?[0],
            norm_input: AffineNorm::bind(file, &format!("{prefix}.shared_block.2"), HIDDEN)?,
            depth_weight: tensor(
                file,
                &format!("{prefix}.shared_block.3.weight"),
                &[HIDDEN, 1, TCN_KERNEL],
            )?,
            depth_bias: tensor(file, &format!("{prefix}.shared_block.3.bias"), &[HIDDEN])?,
            prelu_depth: tensor(file, &format!("{prefix}.shared_block.4.weight"), &[1])?[0],
            norm_depth: AffineNorm::bind(file, &format!("{prefix}.shared_block.5"), HIDDEN)?,
            residual: Conv1d::bind(
                file,
                &format!("{prefix}.res_conv"),
                HIDDEN,
                BOTTLENECK,
                1,
                true,
            )?,
            skip: Conv1d::bind(file, &format!("{prefix}.skip_conv"), HIDDEN, SKIP, 1, true)?,
            dilation: 1 << (index % BLOCKS),
        })
    }

    fn forward(
        &self,
        input: &[f32],
        frames: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let (mut hidden, hidden_frames) = self.input.forward(input, frames, 1, 0, compute)?;
        debug_assert_eq!(hidden_frames, frames);
        prelu_inplace(&mut hidden, self.prelu_input);
        hidden = global_layer_norm(&hidden, HIDDEN, frames, &self.norm_input, compute)?;
        hidden = dilated_depthwise(
            &hidden,
            frames,
            &self.depth_weight,
            &self.depth_bias,
            self.dilation,
            compute,
        )?;
        prelu_inplace(&mut hidden, self.prelu_depth);
        hidden = global_layer_norm(&hidden, HIDDEN, frames, &self.norm_depth, compute)?;
        let (residual, residual_frames) = self.residual.forward(&hidden, frames, 1, 0, compute)?;
        let (skip, skip_frames) = self.skip.forward(&hidden, frames, 1, 0, compute)?;
        debug_assert_eq!(residual_frames, frames);
        debug_assert_eq!(skip_frames, frames);
        Ok((residual, skip))
    }
}

#[derive(Debug)]
struct NetworkWeights {
    encoder: Conv1d,
    bottleneck_norm: AffineNorm,
    bottleneck_conv: Conv1d,
    blocks: Vec<TcnBlock>,
    mask_prelu: f32,
    mask_conv: Conv1d,
    decoder: Vec<f32>,
}

impl NetworkWeights {
    fn bind(file: &GgufFile) -> Result<Self> {
        Ok(Self {
            encoder: Conv1d {
                weight: tensor(
                    file,
                    "encoder.filterbank._filters",
                    &[FILTERS, 1, ENCODER_KERNEL],
                )?,
                bias: None,
                input_channels: 1,
                output_channels: FILTERS,
                kernel: ENCODER_KERNEL,
            },
            bottleneck_norm: AffineNorm::bind(file, "masker.bottleneck.0", FILTERS)?,
            bottleneck_conv: Conv1d::bind(
                file,
                "masker.bottleneck.1",
                FILTERS,
                BOTTLENECK,
                1,
                true,
            )?,
            blocks: (0..BLOCKS * REPEATS)
                .map(|index| TcnBlock::bind(file, index))
                .collect::<Result<Vec<_>>>()?,
            mask_prelu: tensor(file, "masker.mask_net.0.weight", &[1])?[0],
            mask_conv: Conv1d::bind(file, "masker.mask_net.1", SKIP, FILTERS, 1, true)?,
            decoder: tensor(
                file,
                "decoder.filterbank._filters",
                &[FILTERS, 1, ENCODER_KERNEL],
            )?,
        })
    }

    fn encode(&self, pcm: &[f32], compute: &Compute) -> Result<(Vec<f32>, usize)> {
        self.encoder
            .forward(pcm, pcm.len(), ENCODER_STRIDE, 0, compute)
    }

    fn bottleneck(&self, encoded: &[f32], frames: usize, compute: &Compute) -> Result<Vec<f32>> {
        let normalized =
            global_layer_norm(encoded, FILTERS, frames, &self.bottleneck_norm, compute)?;
        self.bottleneck_conv
            .forward(&normalized, frames, 1, 0, compute)
            .map(|(values, _)| values)
    }

    fn mask(&self, encoded: &[f32], frames: usize, compute: &Compute) -> Result<Vec<f32>> {
        let mut hidden = self.bottleneck(encoded, frames, compute)?;
        let mut skip_sum = vec![0.0; SKIP * frames];
        for block in &self.blocks {
            let (residual, skip) = block.forward(&hidden, frames, compute)?;
            add_inplace(&mut hidden, &residual);
            add_inplace(&mut skip_sum, &skip);
        }
        prelu_inplace(&mut skip_sum, self.mask_prelu);
        let (mut mask, mask_frames) = self.mask_conv.forward(&skip_sum, frames, 1, 0, compute)?;
        debug_assert_eq!(mask_frames, frames);
        for value in &mut mask {
            *value = value.max(0.0);
        }
        Ok(mask)
    }

    fn separate(&self, pcm: &[f32], compute: &Compute) -> Result<Vec<Vec<f32>>> {
        let (encoded, frames) = self.encode(pcm, compute)?;
        let mask = self.mask(&encoded, frames, compute)?;
        let masked = encoded
            .iter()
            .zip(mask)
            .map(|(encoded, mask)| encoded * mask)
            .collect::<Vec<_>>();
        let mut decoded = conv_transpose1d(
            &masked,
            FILTERS,
            frames,
            &self.decoder,
            1,
            ENCODER_KERNEL,
            ENCODER_STRIDE,
            compute,
        )?;
        decoded.resize(pcm.len(), 0.0);
        decoded.truncate(pcm.len());
        Ok(vec![decoded])
    }
}

/// Strictly bound official Conv-TasNet tensors.
#[derive(Debug)]
pub struct ConvTasnetWeights {
    network: Box<NetworkWeights>,
    tensor_count: usize,
}

impl ConvTasnetWeights {
    /// Validates and dequantizes the exact 345-tensor official manifest.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        if file.tensors().len() != TENSOR_COUNT {
            return Err(VokraError::ModelLoad(format!(
                "conv_tasnet: GGUF has {} tensors, expected exactly {TENSOR_COUNT}",
                file.tensors().len()
            )));
        }
        Ok(Self {
            network: Box::new(NetworkWeights::bind(file)?),
            tensor_count: file.tensors().len(),
        })
    }

    #[must_use]
    /// Returns the exact number of bound tensors.
    pub fn tensor_count(&self) -> usize {
        self.tensor_count
    }
}

/// Native Conv-TasNet inference handle with explicit CPU/Metal dispatch.
#[derive(Debug)]
pub struct ConvTasnet {
    config: ConvTasnetConfig,
    weights: ConvTasnetWeights,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl ConvTasnet {
    /// Strictly binds a corrected official checkpoint GGUF.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        require_string(file, chunks::KEY_MODEL_NAME, NAME)?;
        require_string(file, KEY_MODEL_CATEGORY, CATEGORY)?;
        require_string(file, KEY_MODEL_ID, NAME)?;
        require_string(file, KEY_UPSTREAM_HF, UPSTREAM_HF)?;
        require_string(file, KEY_UPSTREAM_REVISION, UPSTREAM_REVISION)?;
        let config = ConvTasnetConfig::from_gguf(file)?;
        let weights = ConvTasnetWeights::from_gguf(file)?;
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|value| value.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        Ok(Self {
            config,
            weights,
            weight_license,
            backend: BackendKind::Cpu,
        })
    }

    /// Selects one backend for every declared learned operation.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    #[must_use]
    /// Returns the selected execution backend.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    #[must_use]
    /// Returns the pinned model topology.
    pub const fn config(&self) -> &ConvTasnetConfig {
        &self.config
    }

    #[must_use]
    /// Returns the required waveform sample rate.
    pub const fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    #[must_use]
    /// Returns the number of enhanced output streams.
    pub const fn n_out(&self) -> u32 {
        self.config.n_src
    }

    #[must_use]
    /// Returns the normalized provenance license class.
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    #[must_use]
    /// Returns the exact number of bound tensors.
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Runs the complete official enhancement forward.
    pub fn separate(&self, mixed_pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        validate_pcm(mixed_pcm)?;
        let compute = Compute::for_backend(self.backend, CONV_TASNET_HOT_OPS)?;
        self.weights.network.separate(mixed_pcm, &compute)
    }

    /// Runs only the learned waveform encoder for parity diagnostics.
    pub fn encode_features(&self, mixed_pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        validate_pcm(mixed_pcm)?;
        let compute = Compute::for_backend(self.backend, CONV_TASNET_HOT_OPS)?;
        self.weights.network.encode(mixed_pcm, &compute)
    }

    /// Runs encoder plus the initial global-normalization/bottleneck projection.
    pub fn bottleneck_features(&self, mixed_pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        validate_pcm(mixed_pcm)?;
        let compute = Compute::for_backend(self.backend, CONV_TASNET_HOT_OPS)?;
        let (encoded, frames) = self.weights.network.encode(mixed_pcm, &compute)?;
        let features = self
            .weights
            .network
            .bottleneck(&encoded, frames, &compute)?;
        Ok((features, frames))
    }

    /// Runs encoder plus the complete TCN mask estimator.
    pub fn mask_features(&self, mixed_pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        validate_pcm(mixed_pcm)?;
        let compute = Compute::for_backend(self.backend, CONV_TASNET_HOT_OPS)?;
        let (encoded, frames) = self.weights.network.encode(mixed_pcm, &compute)?;
        let mask = self.weights.network.mask(&encoded, frames, &compute)?;
        Ok((mask, frames))
    }
}

impl vokra_core::engines::SeparationEngine for ConvTasnet {
    fn separate(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        ConvTasnet::separate(self, pcm)
    }

    fn sample_rate(&self) -> u32 {
        ConvTasnet::sample_rate(self)
    }

    fn output_streams(&self) -> usize {
        self.n_out() as usize
    }

    fn backend(&self) -> BackendKind {
        ConvTasnet::backend(self)
    }
}

fn validate_pcm(pcm: &[f32]) -> Result<()> {
    if pcm.len() < ENCODER_KERNEL {
        return Err(VokraError::InvalidArgument(format!(
            "conv_tasnet: input has {} samples, minimum is {ENCODER_KERNEL}",
            pcm.len()
        )));
    }
    if let Some(index) = pcm.iter().position(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "conv_tasnet: input sample {index} is not finite"
        )));
    }
    Ok(())
}

fn global_layer_norm(
    input: &[f32],
    channels: usize,
    frames: usize,
    norm: &AffineNorm,
    compute: &Compute,
) -> Result<Vec<f32>> {
    if input.len() != channels * frames
        || norm.gamma.len() != channels
        || norm.beta.len() != channels
    {
        return Err(VokraError::InvalidArgument(
            "conv_tasnet: global LayerNorm shape mismatch".to_owned(),
        ));
    }
    let mut gamma = Vec::with_capacity(input.len());
    let mut beta = Vec::with_capacity(input.len());
    for channel in 0..channels {
        gamma.extend(std::iter::repeat_n(norm.gamma[channel], frames));
        beta.extend(std::iter::repeat_n(norm.beta[channel], frames));
    }
    let mut output = vec![0.0; input.len()];
    compute.layer_norm_f32(input, &mut output, 1, input.len(), &gamma, &beta, GLN_EPS)?;
    Ok(output)
}

fn dilated_depthwise(
    input: &[f32],
    frames: usize,
    weight: &[f32],
    bias: &[f32],
    dilation: usize,
    compute: &Compute,
) -> Result<Vec<f32>> {
    let effective_kernel = (TCN_KERNEL - 1) * dilation + 1;
    let mut expanded = vec![0.0; HIDDEN * effective_kernel];
    for channel in 0..HIDDEN {
        for tap in 0..TCN_KERNEL {
            expanded[channel * effective_kernel + tap * dilation] =
                weight[channel * TCN_KERNEL + tap];
        }
    }
    let mut output = vec![0.0; HIDDEN * frames];
    compute.grouped_conv1d_f32(
        input,
        HIDDEN,
        frames,
        &expanded,
        HIDDEN,
        effective_kernel,
        Some(bias),
        1,
        dilation,
        HIDDEN,
        &mut output,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose1d(
    input: &[f32],
    input_channels: usize,
    input_length: usize,
    weight: &[f32],
    output_channels: usize,
    kernel: usize,
    stride: usize,
    compute: &Compute,
) -> Result<Vec<f32>> {
    if input_length == 0 || stride == 0 || kernel == 0 {
        return Err(VokraError::InvalidArgument(
            "conv_tasnet: invalid ConvTranspose1D extent".to_owned(),
        ));
    }
    let expanded_length = (input_length - 1) * stride + 1;
    let mut expanded_input = vec![0.0; input_channels * expanded_length];
    for channel in 0..input_channels {
        for frame in 0..input_length {
            expanded_input[channel * expanded_length + frame * stride] =
                input[channel * input_length + frame];
        }
    }
    let mut flipped = vec![0.0; output_channels * input_channels * kernel];
    for input_channel in 0..input_channels {
        for output_channel in 0..output_channels {
            for tap in 0..kernel {
                flipped[(output_channel * input_channels + input_channel) * kernel + tap] = weight
                    [(input_channel * output_channels + output_channel) * kernel + kernel
                        - 1
                        - tap];
            }
        }
    }
    let padding = kernel - 1;
    let output_length = expanded_length + 2 * padding - kernel + 1;
    let mut output = vec![0.0; output_channels * output_length];
    compute.conv1d_f32(
        &expanded_input,
        input_channels,
        expanded_length,
        &flipped,
        output_channels,
        kernel,
        None,
        1,
        padding,
        &mut output,
    )?;
    Ok(output)
}

fn prelu_inplace(values: &mut [f32], slope: f32) {
    for value in values {
        if *value < 0.0 {
            *value *= slope;
        }
    }
}

fn add_inplace(destination: &mut [f32], source: &[f32]) {
    debug_assert_eq!(destination.len(), source.len());
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += source;
    }
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("conv_tasnet: missing tensor `{name}`")))?;
    let expected = expected
        .iter()
        .map(|&dimension| dimension as u64)
        .collect::<Vec<_>>();
    if info.dimensions != expected {
        return Err(VokraError::ModelLoad(format!(
            "conv_tasnet: tensor `{name}` has dims {:?}, expected {expected:?}",
            info.dimensions
        )));
    }
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("conv_tasnet: reading tensor `{name}`: {error}"))
    })
}

fn required_u32(file: &GgufFile, key: &str) -> Result<u32> {
    match file.get(key) {
        Some(vokra_core::gguf::GgufMetadataValue::U32(value)) => Ok(*value),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "conv_tasnet: `{key}` must be U32, got {other:?}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "conv_tasnet: missing required topology metadata `{key}`"
        ))),
    }
}

fn required_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| VokraError::ModelLoad(format!("conv_tasnet: missing string `{key}`")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = required_string(file, key)?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "conv_tasnet: `{key}` is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_topology_is_pinned() {
        let config = ConvTasnetConfig::default();
        assert_eq!(config.n_filters, 512);
        assert_eq!(config.n_kernel, 32);
        assert_eq!(config.stride, 16);
        assert_eq!(config.n_blocks * config.n_repeats, 24);
        assert_eq!(config.n_src, 1);
        assert_eq!(config.sample_rate, 16_000);
    }

    #[test]
    fn prelu_uses_one_learned_slope() {
        let mut values = [-2.0, 0.0, 3.0];
        prelu_inplace(&mut values, 0.25);
        assert_eq!(values, [-0.5, 0.0, 3.0]);
    }

    #[test]
    fn too_short_and_non_finite_pcm_fail_loudly() {
        assert!(validate_pcm(&[0.0; ENCODER_KERNEL - 1]).is_err());
        let mut pcm = [0.0; ENCODER_KERNEL];
        pcm[4] = f32::NAN;
        assert!(validate_pcm(&pcm).is_err());
    }
}
