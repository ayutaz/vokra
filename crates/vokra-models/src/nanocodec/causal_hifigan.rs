//! Stateful causal HiFi-GAN decoder used by NVIDIA NanoCodec.
//!
//! # Primary-source contract
//!
//! The topology and causal padding rules are transcribed from NVIDIA NeMo
//! Speech's Apache-2.0 `CausalHiFiGANDecoder`, `CausalConv1dNorm`,
//! `CausalConvTranspose1dNorm`, `HiFiGANResLayer`, and `HalfSnake` at pinned
//! commit `4fcff72febec9395fdbd4bfa0747bfda2ecd3cef`. Weight naming and the
//! grouped-to-dense ConvTranspose transform are cross-checked against NVIDIA's
//! Apache-2.0 NeMo-Speech.cpp converter/runtime at pinned commit
//! `4f9676226f667d14608487df744f375db87127f8`.
//!
//! The implementation carries no NVIDIA weights. Checkpoint license and
//! publication eligibility remain separate fail-closed converter/model-zoo
//! decisions; adding this decoder does not authorize redistribution.

use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

use crate::compute::Compute;
use crate::mimi::nn::{CausalConv1d, CausalConvTranspose1d, ConvState, ConvTrState};

const HALF_SNAKE_EPSILON: f32 = 1.0e-9;
const LEAKY_RELU_SLOPE: f32 = 0.01;
const NEMO_SPEECH_COMMIT: &str = "4fcff72febec9395fdbd4bfa0747bfda2ecd3cef";
const NEMO_SOURCE_URL: &str = "https://github.com/NVIDIA-NeMo/Speech.git";
const PROFILE_06: &str = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps";
const PROFILE_178: &str = "nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps";
const PROFILE_189: &str = "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps";
const REVISION_06: &str = "5c8e22ed763c14d81337fbe6ca74062f3d10f7e5";
const REVISION_178: &str = "c4ab84a92c8d36a8b5a79eaea807cfaf7f03ed86";
const REVISION_189: &str = "fc00890b604aa2de298d2641ffc6c5f6caf8c4d7";

/// Checkpoint-derived geometry of the NanoCodec causal HiFi-GAN decoder.
///
/// No public NanoCodec profile is baked into this type. In particular,
/// [`frame_hop`](Self::frame_hop) and [`upsample_rates`](Self::upsample_rates)
/// are independently read from checkpoint metadata and then cross-checked.
/// A mismatch is rejected: dropping or resampling generator output would
/// invent behavior that is absent from the NeMo reference forward pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalHifiGanConfig {
    /// Per-frame latent feature width.
    pub input_dim: usize,
    /// Output channels of the causal pre-convolution.
    pub base_channels: usize,
    /// PCM samples emitted per input frame.
    pub frame_hop: usize,
    /// Per-stage transposed-convolution strides.
    pub upsample_rates: Vec<usize>,
    /// Pre-convolution kernel size.
    pub input_kernel_size: usize,
    /// PCM projection kernel size.
    pub output_kernel_size: usize,
    /// Parallel residual-branch kernel sizes.
    pub resblock_kernel_sizes: Vec<usize>,
    /// Sequential dilation values inside every residual branch.
    pub resblock_dilations: Vec<usize>,
}

impl CausalHifiGanConfig {
    /// Validates all shape/rate invariants without assuming a known model
    /// variant.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] for zero dimensions, overflow,
    /// inconsistent channel halving, or a `frame_hop` that differs from the
    /// causal generator's stride product.
    pub fn validate(&self) -> Result<()> {
        if self.input_dim == 0
            || self.base_channels == 0
            || self.frame_hop == 0
            || self.input_kernel_size == 0
            || self.output_kernel_size == 0
        {
            return Err(VokraError::InvalidArgument(
                "nanocodec causal HiFi-GAN config: dimensions, kernels, and frame_hop must be > 0"
                    .into(),
            ));
        }
        if self.upsample_rates.is_empty()
            || self.resblock_kernel_sizes.is_empty()
            || self.resblock_dilations.is_empty()
        {
            return Err(VokraError::InvalidArgument(
                "nanocodec causal HiFi-GAN config: upsample rates, residual kernels, and residual dilations must be non-empty"
                    .into(),
            ));
        }
        if self.upsample_rates.contains(&0)
            || self.resblock_kernel_sizes.contains(&0)
            || self.resblock_dilations.contains(&0)
        {
            return Err(VokraError::InvalidArgument(
                "nanocodec causal HiFi-GAN config: rates, residual kernels, and dilations must be > 0"
                    .into(),
            ));
        }
        let hop = self
            .upsample_rates
            .iter()
            .try_fold(1usize, |product, &rate| {
                product.checked_mul(rate).ok_or_else(|| {
                    VokraError::InvalidArgument(
                        "nanocodec causal HiFi-GAN config: upsample-rate product overflow".into(),
                    )
                })
            })?;
        if self.frame_hop != hop {
            return Err(VokraError::InvalidArgument(format!(
                "nanocodec causal HiFi-GAN config: checkpoint frame_hop {} != generator product(upsample_rates) {hop}; refusing to drop or invent waveform samples",
                self.frame_hop
            )));
        }
        let mut channels = self.base_channels;
        for (stage, &rate) in self.upsample_rates.iter().enumerate() {
            if channels < 2 || channels % 2 != 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "nanocodec causal HiFi-GAN config: stage {stage} input channels {channels} cannot be halved"
                )));
            }
            let _kernel = rate.checked_mul(2).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "nanocodec causal HiFi-GAN config: stage {stage} upsample kernel overflow"
                ))
            })?;
            channels /= 2;
        }
        Ok(())
    }
}

/// Dense causal Conv1d weights in `[out_channels, in_channels, kernel]`
/// row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalHifiGanConv1dWeights {
    /// Weight tensor flattened in OIK order.
    pub weight: Vec<f32>,
    /// Bias vector with one value per output channel.
    pub bias: Vec<f32>,
}

/// Dense causal ConvTranspose1d weights in PyTorch's
/// `[in_channels, out_channels, kernel]` row-major order.
///
/// NanoCodec checkpoints use `groups == out_channels`. The offline converter
/// expands that sparse grouped tensor to this dense form once, exactly as the
/// official NeMo-Speech.cpp converter does; runtime decoding then has no
/// grouping branch and performs no allocation.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalHifiGanConvTranspose1dWeights {
    /// Weight tensor flattened in IOK order.
    pub weight: Vec<f32>,
    /// Bias vector with one value per output channel.
    pub bias: Vec<f32>,
}

/// Learned parameters of one HalfSnake activation.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalHifiGanHalfSnakeWeights {
    /// Snake alpha for the first `channels / 2` channels.
    pub alpha: Vec<f32>,
    /// Precomputed `1 / (alpha + 1e-9)` for the same channels.
    pub alpha_inv: Vec<f32>,
}

/// One sequential residual block inside a kernel-size branch.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalHifiGanResidualBlockWeights {
    /// Activation before the dilated input convolution.
    pub input_activation: CausalHifiGanHalfSnakeWeights,
    /// Dilated causal convolution.
    pub input_conv: CausalHifiGanConv1dWeights,
    /// Activation before the unit-dilation skip convolution.
    pub skip_activation: CausalHifiGanHalfSnakeWeights,
    /// Unit-dilation causal convolution.
    pub skip_conv: CausalHifiGanConv1dWeights,
    /// Dilation of `input_conv`, repeated here to fail closed if tensor order
    /// disagrees with checkpoint metadata.
    pub dilation: usize,
}

/// Weights of one upsample stage.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalHifiGanStageWeights {
    /// HalfSnake before the transposed convolution.
    pub activation: CausalHifiGanHalfSnakeWeights,
    /// Causal right-trim transposed convolution.
    pub upsample: CausalHifiGanConvTranspose1dWeights,
    /// One branch per configured residual kernel, each containing one block
    /// per configured dilation.
    pub residual_branches: Vec<Vec<CausalHifiGanResidualBlockWeights>>,
}

/// Complete decoder weight set.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalHifiGanWeights {
    /// Input projection.
    pub pre_conv: CausalHifiGanConv1dWeights,
    /// Upsample stages.
    pub stages: Vec<CausalHifiGanStageWeights>,
    /// Final HalfSnake activation.
    pub post_activation: CausalHifiGanHalfSnakeWeights,
    /// Mono PCM projection.
    pub post_conv: CausalHifiGanConv1dWeights,
}

#[derive(Debug, Clone)]
struct HalfSnake {
    channels: usize,
    alpha: Vec<f32>,
    alpha_inv: Vec<f32>,
}

impl HalfSnake {
    fn new(channels: usize, weights: CausalHifiGanHalfSnakeWeights, context: &str) -> Result<Self> {
        let snake_channels = channels / 2;
        if weights.alpha.len() != snake_channels || weights.alpha_inv.len() != snake_channels {
            return Err(VokraError::InvalidArgument(format!(
                "nanocodec causal HiFi-GAN {context}: HalfSnake alpha/alpha_inv lengths ({}/{}) != channels/2 {snake_channels}",
                weights.alpha.len(),
                weights.alpha_inv.len()
            )));
        }
        for (channel, (&alpha, &alpha_inv)) in weights
            .alpha
            .iter()
            .zip(weights.alpha_inv.iter())
            .enumerate()
        {
            let expected = 1.0 / (alpha + HALF_SNAKE_EPSILON);
            let scale = expected.abs().max(1.0);
            if !alpha.is_finite()
                || !alpha_inv.is_finite()
                || !expected.is_finite()
                || (alpha_inv - expected).abs() > 1.0e-6 * scale
            {
                return Err(VokraError::InvalidArgument(format!(
                    "nanocodec causal HiFi-GAN {context}: HalfSnake alpha_inv[{channel}] is not 1/(alpha+1e-9)"
                )));
            }
        }
        Ok(Self {
            channels,
            alpha: weights.alpha,
            alpha_inv: weights.alpha_inv,
        })
    }

    fn apply_inplace(&self, values: &mut [f32], time: usize) {
        debug_assert_eq!(values.len(), self.channels * time);
        let snake_channels = self.channels / 2;
        for channel in 0..snake_channels {
            let alpha = self.alpha[channel];
            let alpha_inv = self.alpha_inv[channel];
            for value in &mut values[channel * time..(channel + 1) * time] {
                let sine = vokra_math::sin(alpha * *value);
                *value += alpha_inv * sine * sine;
            }
        }
        for channel in snake_channels..self.channels {
            for value in &mut values[channel * time..(channel + 1) * time] {
                if *value < 0.0 {
                    *value *= LEAKY_RELU_SLOPE;
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ResidualBlock {
    input_activation: HalfSnake,
    input_conv: CausalConv1d,
    skip_activation: HalfSnake,
    skip_conv: CausalConv1d,
}

#[derive(Debug, Clone)]
struct DecoderStage {
    activation: HalfSnake,
    upsample: CausalConvTranspose1d,
    residual_branches: Vec<Vec<ResidualBlock>>,
}

/// Stateful, CPU-native NanoCodec causal HiFi-GAN decoder.
///
/// `decode_into` consumes row-major `[frames, input_dim]` features, emits
/// exactly `frames * frame_hop` samples, and carries every causal convolution
/// history and transposed-convolution overlap tail in [`CausalHifiGanState`].
/// `frame_hop` must equal the causal generator's checked upsample-rate
/// product, so every reference waveform sample is emitted exactly once.
pub struct CausalHifiGan {
    config: CausalHifiGanConfig,
    pre_conv: CausalConv1d,
    stages: Vec<DecoderStage>,
    post_activation: HalfSnake,
    post_conv: CausalConv1d,
    geometry_signature: u64,
}

impl std::fmt::Debug for CausalHifiGan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CausalHifiGan")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
struct ResidualBlockState {
    input_conv: ConvState,
    skip_conv: ConvState,
}

#[derive(Debug, Clone)]
struct DecoderStageState {
    upsample: ConvTrState,
    residual_branches: Vec<Vec<ResidualBlockState>>,
    work: Vec<f32>,
    base: Vec<f32>,
    mid: Vec<f32>,
    sum: Vec<f32>,
}

/// Preallocated state and scratch for one causal HiFi-GAN stream.
///
/// Construct it with [`CausalHifiGan::state`]. Its fields are intentionally
/// private so state from a different decoder geometry cannot be fabricated.
pub struct CausalHifiGanState {
    geometry_signature: u64,
    frames_cap: usize,
    input: Vec<f32>,
    pre_conv: ConvState,
    stages: Vec<DecoderStageState>,
    post_conv: ConvState,
    edges: Vec<Vec<f32>>,
    raw_pcm: Vec<f32>,
}

impl std::fmt::Debug for CausalHifiGanState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CausalHifiGanState")
            .field("frames_cap", &self.frames_cap)
            .finish_non_exhaustive()
    }
}

impl CausalHifiGan {
    /// Binds validated checkpoint geometry and weights.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] for a config invariant, tensor shape,
    /// stage/branch count, or dilation-order mismatch.
    pub fn new(config: CausalHifiGanConfig, weights: CausalHifiGanWeights) -> Result<Self> {
        config.validate()?;
        if weights.stages.len() != config.upsample_rates.len() {
            return Err(VokraError::InvalidArgument(format!(
                "nanocodec causal HiFi-GAN: {} weight stages != {} configured upsample rates",
                weights.stages.len(),
                config.upsample_rates.len()
            )));
        }

        let pre_conv = bind_conv(
            config.input_dim,
            config.base_channels,
            config.input_kernel_size,
            1,
            weights.pre_conv,
            "pre_conv",
        )?;
        let mut in_channels = config.base_channels;
        let mut stages = Vec::with_capacity(weights.stages.len());
        for (stage_index, (stage_weights, &rate)) in weights
            .stages
            .into_iter()
            .zip(config.upsample_rates.iter())
            .enumerate()
        {
            let out_channels = in_channels / 2;
            let up_kernel = rate.checked_mul(2).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "nanocodec causal HiFi-GAN stage {stage_index}: upsample kernel overflow"
                ))
            })?;
            let activation = HalfSnake::new(
                in_channels,
                stage_weights.activation,
                &format!("stage {stage_index} activation"),
            )?;
            let upsample = bind_conv_transpose(
                in_channels,
                out_channels,
                up_kernel,
                rate,
                stage_weights.upsample,
                &format!("stage {stage_index} upsample"),
            )?;
            if stage_weights.residual_branches.len() != config.resblock_kernel_sizes.len() {
                return Err(VokraError::InvalidArgument(format!(
                    "nanocodec causal HiFi-GAN stage {stage_index}: {} residual branches != {} configured kernels",
                    stage_weights.residual_branches.len(),
                    config.resblock_kernel_sizes.len()
                )));
            }
            let mut branches = Vec::with_capacity(stage_weights.residual_branches.len());
            for (branch_index, (blocks, &kernel)) in stage_weights
                .residual_branches
                .into_iter()
                .zip(config.resblock_kernel_sizes.iter())
                .enumerate()
            {
                if blocks.len() != config.resblock_dilations.len() {
                    return Err(VokraError::InvalidArgument(format!(
                        "nanocodec causal HiFi-GAN stage {stage_index} branch {branch_index}: {} blocks != {} configured dilations",
                        blocks.len(),
                        config.resblock_dilations.len()
                    )));
                }
                let mut bound_blocks = Vec::with_capacity(blocks.len());
                for (block_index, (block, &dilation)) in blocks
                    .into_iter()
                    .zip(config.resblock_dilations.iter())
                    .enumerate()
                {
                    if block.dilation != dilation {
                        return Err(VokraError::InvalidArgument(format!(
                            "nanocodec causal HiFi-GAN stage {stage_index} branch {branch_index} block {block_index}: weight dilation {} != configured {dilation}",
                            block.dilation
                        )));
                    }
                    let context =
                        format!("stage {stage_index} branch {branch_index} block {block_index}");
                    bound_blocks.push(ResidualBlock {
                        input_activation: HalfSnake::new(
                            out_channels,
                            block.input_activation,
                            &format!("{context} input activation"),
                        )?,
                        input_conv: bind_conv(
                            out_channels,
                            out_channels,
                            kernel,
                            dilation,
                            block.input_conv,
                            &format!("{context} input_conv"),
                        )?,
                        skip_activation: HalfSnake::new(
                            out_channels,
                            block.skip_activation,
                            &format!("{context} skip activation"),
                        )?,
                        skip_conv: bind_conv(
                            out_channels,
                            out_channels,
                            kernel,
                            1,
                            block.skip_conv,
                            &format!("{context} skip_conv"),
                        )?,
                    });
                }
                branches.push(bound_blocks);
            }
            stages.push(DecoderStage {
                activation,
                upsample,
                residual_branches: branches,
            });
            in_channels = out_channels;
        }
        let post_activation =
            HalfSnake::new(in_channels, weights.post_activation, "post activation")?;
        let post_conv = bind_conv(
            in_channels,
            1,
            config.output_kernel_size,
            1,
            weights.post_conv,
            "post_conv",
        )?;
        let geometry_signature = geometry_signature(&config);
        Ok(Self {
            config,
            pre_conv,
            stages,
            post_activation,
            post_conv,
            geometry_signature,
        })
    }

    /// Binds the exact decoder-only GGUF schema emitted by
    /// `vokra-convert --model nanocodec`.
    ///
    /// Metadata and every tensor shape are checked before construction.  The
    /// loader also rejects a checkpoint whose stamped frame hop differs from
    /// the independently stamped generator stride product; it never drops or
    /// invents waveform samples to make such a checkpoint appear usable.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] for a foreign architecture, missing or
    /// malformed metadata, unsupported transform markers, missing/non-F32
    /// tensors, shape mismatches, or invalid bound decoder geometry.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_metadata_string(file, "vokra.model.arch", "nanocodec")?;
        require_metadata_string(file, "vokra.nanocodec.activation", "HalfSnake")?;
        require_metadata_string(file, "vokra.nanocodec.output_activation", "ClampActivation")?;
        require_metadata_string(file, "vokra.nanocodec.pad_mode", "zeros")?;
        require_metadata_string(
            file,
            "vokra.nanocodec.nemo_speech_commit",
            NEMO_SPEECH_COMMIT,
        )?;
        require_metadata_string(file, "vokra.nanocodec.nemo_source_url", NEMO_SOURCE_URL)?;
        require_metadata_bool(file, "vokra.nanocodec.grouped_upsample_expanded", true)?;

        let n_codebooks = require_metadata_usize(file, "vokra.nanocodec.n_codebooks")?;
        let levels_per_group =
            require_metadata_usize_array(file, "vokra.nanocodec.levels_per_group")?;
        let input_dim = require_metadata_usize(file, "vokra.nanocodec.embed_dim")?;
        let sample_rate = require_metadata_usize(file, "vokra.nanocodec.sample_rate")?;
        let frame_hop = require_metadata_usize(file, "vokra.nanocodec.frame_hop")?;
        let generator_hop = require_metadata_usize(file, "vokra.nanocodec.generator_hop")?;
        let base_channels = require_metadata_usize(file, "vokra.nanocodec.base_channels")?;
        let upsample_rates = require_metadata_usize_array(file, "vokra.nanocodec.upsample_rates")?;
        let input_kernel_size = require_metadata_usize(file, "vokra.nanocodec.input_kernel_size")?;
        let output_kernel_size =
            require_metadata_usize(file, "vokra.nanocodec.output_kernel_size")?;
        let resblock_kernel_sizes =
            require_metadata_usize_array(file, "vokra.nanocodec.resblock_kernel_sizes")?;
        let resblock_dilations =
            require_metadata_usize_array(file, "vokra.nanocodec.resblock_dilations")?;
        let source_model_id =
            require_metadata_nonempty_string(file, "vokra.provenance.upstream_hf")?;
        let source_revision =
            require_metadata_nonempty_string(file, "vokra.provenance.upstream_revision")?;
        let checkpoint_sha256 =
            require_metadata_nonempty_string(file, "vokra.provenance.checkpoint_sha256")?;

        if n_codebooks == 0
            || levels_per_group.is_empty()
            || levels_per_group.iter().any(|&level| level < 2)
            || sample_rate == 0
        {
            return Err(load_error(
                "n_codebooks/sample_rate must be non-zero and levels_per_group entries must be >= 2"
                    .to_owned(),
            ));
        }
        let expected_input_dim = n_codebooks
            .checked_mul(levels_per_group.len())
            .ok_or_else(|| load_error("codebook/group dimension overflow".to_owned()))?;
        if input_dim != expected_input_dim {
            return Err(load_error(format!(
                "embed_dim {input_dim} != n_codebooks {n_codebooks} * levels_per_group length {}",
                levels_per_group.len()
            )));
        }
        if frame_hop != generator_hop {
            return Err(load_error(format!(
                "checkpoint frame_hop {frame_hop} != stamped generator_hop {generator_hop}; refusing to drop or invent waveform samples"
            )));
        }
        validate_audited_profile_metadata(
            &source_model_id,
            &source_revision,
            &checkpoint_sha256,
            sample_rate,
            base_channels,
            n_codebooks,
            &levels_per_group,
            input_dim,
            frame_hop,
            &upsample_rates,
        )?;

        let config = CausalHifiGanConfig {
            input_dim,
            base_channels,
            frame_hop,
            upsample_rates,
            input_kernel_size,
            output_kernel_size,
            resblock_kernel_sizes,
            resblock_dilations,
        };
        config
            .validate()
            .map_err(|error| load_error(format!("invalid checkpoint geometry: {error}")))?;
        let pre_conv = load_conv1d(
            file,
            "nanocodec.pre_conv",
            config.base_channels,
            config.input_dim,
            config.input_kernel_size,
        )?;
        let mut in_channels = config.base_channels;
        let mut stages = Vec::with_capacity(config.upsample_rates.len());
        for (stage_index, &rate) in config.upsample_rates.iter().enumerate() {
            let out_channels = in_channels / 2;
            let prefix = format!("nanocodec.stage.{stage_index}");
            let activation = load_half_snake(file, &format!("{prefix}.activation"), in_channels)?;
            let upsample = CausalHifiGanConvTranspose1dWeights {
                weight: load_f32_tensor(
                    file,
                    &format!("{prefix}.upsample.weight"),
                    &[in_channels, out_channels, rate * 2],
                )?,
                bias: load_f32_tensor(file, &format!("{prefix}.upsample.bias"), &[out_channels])?,
            };
            let mut residual_branches = Vec::with_capacity(config.resblock_kernel_sizes.len());
            for (branch_index, &kernel) in config.resblock_kernel_sizes.iter().enumerate() {
                let mut blocks = Vec::with_capacity(config.resblock_dilations.len());
                for (block_index, &dilation) in config.resblock_dilations.iter().enumerate() {
                    let block_prefix =
                        format!("{prefix}.branch.{branch_index}.block.{block_index}");
                    blocks.push(CausalHifiGanResidualBlockWeights {
                        input_activation: load_half_snake(
                            file,
                            &format!("{block_prefix}.input_activation"),
                            out_channels,
                        )?,
                        input_conv: load_conv1d(
                            file,
                            &format!("{block_prefix}.input_conv"),
                            out_channels,
                            out_channels,
                            kernel,
                        )?,
                        skip_activation: load_half_snake(
                            file,
                            &format!("{block_prefix}.skip_activation"),
                            out_channels,
                        )?,
                        skip_conv: load_conv1d(
                            file,
                            &format!("{block_prefix}.skip_conv"),
                            out_channels,
                            out_channels,
                            kernel,
                        )?,
                        dilation,
                    });
                }
                residual_branches.push(blocks);
            }
            stages.push(CausalHifiGanStageWeights {
                activation,
                upsample,
                residual_branches,
            });
            in_channels = out_channels;
        }
        let weights = CausalHifiGanWeights {
            pre_conv,
            stages,
            post_activation: load_half_snake(file, "nanocodec.post_activation", in_channels)?,
            post_conv: load_conv1d(
                file,
                "nanocodec.post_conv",
                1,
                in_channels,
                config.output_kernel_size,
            )?,
        };
        Self::new(config, weights)
            .map_err(|error| load_error(format!("invalid bound decoder: {error}")))
    }

    /// Validated checkpoint geometry.
    #[must_use]
    pub fn config(&self) -> &CausalHifiGanConfig {
        &self.config
    }

    /// Per-frame feature width.
    #[must_use]
    pub fn expected_feature_dim(&self) -> usize {
        self.config.input_dim
    }

    /// PCM samples emitted for each feature frame.
    ///
    /// This value is read from checkpoint metadata rather than hardcoded, and
    /// construction verifies it against the independently read
    /// transposed-convolution stride product.
    #[must_use]
    pub fn frame_hop(&self) -> usize {
        self.config.frame_hop
    }

    /// Creates preallocated stream state for at most `frames_cap` frames per
    /// `decode_into` call.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] for zero capacity or size overflow.
    pub fn state(&self, frames_cap: usize) -> Result<CausalHifiGanState> {
        if frames_cap == 0 {
            return Err(VokraError::InvalidArgument(
                "nanocodec causal HiFi-GAN state: frames_cap must be > 0".into(),
            ));
        }
        let input_len = checked_area(self.config.input_dim, frames_cap, "state input scratch")?;
        let mut edges = Vec::with_capacity(self.stages.len() + 1);
        let mut time = frames_cap;
        let mut channels = self.config.base_channels;
        edges.push(vec![
            0.0;
            checked_area(channels, time, "state pre-conv edge")?
        ]);
        let mut stage_states = Vec::with_capacity(self.stages.len());
        for (stage_index, stage) in self.stages.iter().enumerate() {
            let upsample = stage.upsample.state(time);
            time = time.checked_mul(stage.upsample.stride).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "nanocodec causal HiFi-GAN state: stage {stage_index} time overflow"
                ))
            })?;
            channels = stage.upsample.out_ch;
            let scratch_len = checked_area(
                channels,
                time,
                &format!("state stage {stage_index} scratch"),
            )?;
            edges.push(vec![0.0; scratch_len]);
            let mut residual_branches = Vec::with_capacity(stage.residual_branches.len());
            for branch in &stage.residual_branches {
                let mut block_states = Vec::with_capacity(branch.len());
                for block in branch {
                    block_states.push(ResidualBlockState {
                        input_conv: block.input_conv.state(time),
                        skip_conv: block.skip_conv.state(time),
                    });
                }
                residual_branches.push(block_states);
            }
            stage_states.push(DecoderStageState {
                upsample,
                residual_branches,
                work: vec![0.0; scratch_len],
                base: vec![0.0; scratch_len],
                mid: vec![0.0; scratch_len],
                sum: vec![0.0; scratch_len],
            });
        }
        Ok(CausalHifiGanState {
            geometry_signature: self.geometry_signature,
            frames_cap,
            input: vec![0.0; input_len],
            pre_conv: self.pre_conv.state(frames_cap),
            stages: stage_states,
            post_conv: self.post_conv.state(time),
            edges,
            raw_pcm: vec![0.0; time],
        })
    }

    /// Clears all carried convolution and overlap state without allocating.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when `state` belongs to a decoder with
    /// different geometry.
    pub fn reset(&self, state: &mut CausalHifiGanState) -> Result<()> {
        self.validate_state(state)?;
        state.pre_conv.reset();
        for stage in &mut state.stages {
            stage.upsample.reset();
            for branch in &mut stage.residual_branches {
                for block in branch {
                    block.input_conv.reset();
                    block.skip_conv.reset();
                }
            }
        }
        state.post_conv.reset();
        Ok(())
    }

    /// Decodes row-major `[frames, input_dim]` features into PCM while carrying
    /// per-layer causal state. The successful steady-state path performs no
    /// heap allocation.
    ///
    /// # Returns
    ///
    /// The number of PCM samples written, always `frames * frame_hop`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] for an empty/misaligned feature buffer,
    /// state mismatch, capacity overflow, or wrong PCM output length.
    pub fn decode_into(
        &self,
        state: &mut CausalHifiGanState,
        features: &[f32],
        pcm_out: &mut [f32],
    ) -> Result<usize> {
        // ZERO-ALLOC-BEGIN — validated steady-state decode uses only state scratch.
        self.validate_state(state)?;
        let feature_dim = self.config.input_dim;
        if features.is_empty() || features.len() % feature_dim != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "nanocodec causal HiFi-GAN decode: feature length {} is not a positive multiple of input_dim {feature_dim}",
                features.len()
            )));
        }
        let frames = features.len() / feature_dim;
        if frames > state.frames_cap {
            return Err(VokraError::InvalidArgument(format!(
                "nanocodec causal HiFi-GAN decode: {frames} frames exceed state capacity {}",
                state.frames_cap
            )));
        }
        let pcm_len = frames.checked_mul(self.config.frame_hop).ok_or_else(|| {
            VokraError::InvalidArgument(
                "nanocodec causal HiFi-GAN decode: PCM length overflow".into(),
            )
        })?;
        if pcm_out.len() != pcm_len {
            return Err(VokraError::InvalidArgument(format!(
                "nanocodec causal HiFi-GAN decode: pcm_out length {} != frames*frame_hop {pcm_len}",
                pcm_out.len()
            )));
        }

        // Public features are frame-major. Causal conv kernels consume
        // channel-major `[channels, time]` buffers.
        for frame in 0..frames {
            for channel in 0..feature_dim {
                state.input[channel * frames + frame] = features[frame * feature_dim + channel];
            }
        }

        let compute = Compute::cpu();
        self.pre_conv.process_into(
            &compute,
            &mut state.pre_conv,
            &state.input[..feature_dim * frames],
            frames,
            &mut state.edges[0][..self.config.base_channels * frames],
        )?;

        let mut time = frames;
        let mut channels = self.config.base_channels;
        for (stage_index, stage) in self.stages.iter().enumerate() {
            stage
                .activation
                .apply_inplace(&mut state.edges[stage_index][..channels * time], time);
            let next_time = time * stage.upsample.stride;
            let next_channels = stage.upsample.out_ch;
            {
                let (before, after) = state.edges.split_at_mut(stage_index + 1);
                stage.upsample.process_into(
                    &compute,
                    &mut state.stages[stage_index].upsample,
                    &before[stage_index][..channels * time],
                    time,
                    &mut after[0][..next_channels * next_time],
                )?;
            }

            let edge_len = next_channels * next_time;
            let edge = &mut state.edges[stage_index + 1][..edge_len];
            let stage_state = &mut state.stages[stage_index];
            stage_state.sum[..edge_len].fill(0.0);
            for (branch_index, branch) in stage.residual_branches.iter().enumerate() {
                stage_state.work[..edge_len].copy_from_slice(edge);
                for (block_index, block) in branch.iter().enumerate() {
                    stage_state.base[..edge_len].copy_from_slice(&stage_state.work[..edge_len]);
                    block
                        .input_activation
                        .apply_inplace(&mut stage_state.work[..edge_len], next_time);
                    block.input_conv.process_into(
                        &compute,
                        &mut stage_state.residual_branches[branch_index][block_index].input_conv,
                        &stage_state.work[..edge_len],
                        next_time,
                        &mut stage_state.mid[..edge_len],
                    )?;
                    block
                        .skip_activation
                        .apply_inplace(&mut stage_state.mid[..edge_len], next_time);
                    block.skip_conv.process_into(
                        &compute,
                        &mut stage_state.residual_branches[branch_index][block_index].skip_conv,
                        &stage_state.mid[..edge_len],
                        next_time,
                        &mut stage_state.work[..edge_len],
                    )?;
                    for i in 0..edge_len {
                        stage_state.work[i] += stage_state.base[i];
                    }
                }
                for i in 0..edge_len {
                    stage_state.sum[i] += stage_state.work[i];
                }
            }
            let reciprocal_branches = 1.0 / stage.residual_branches.len() as f32;
            for (sample, &sum) in edge[..edge_len]
                .iter_mut()
                .zip(&stage_state.sum[..edge_len])
            {
                *sample = sum * reciprocal_branches;
            }
            time = next_time;
            channels = next_channels;
        }

        let final_edge = &mut state.edges[self.stages.len()];
        self.post_activation
            .apply_inplace(&mut final_edge[..channels * time], time);
        debug_assert_eq!(time, pcm_len);
        self.post_conv.process_into(
            &compute,
            &mut state.post_conv,
            &final_edge[..channels * time],
            time,
            &mut state.raw_pcm[..pcm_len],
        )?;
        for (sample, &raw) in pcm_out.iter_mut().zip(&state.raw_pcm[..pcm_len]) {
            *sample = raw.clamp(-1.0, 1.0);
        }
        // ZERO-ALLOC-END
        Ok(pcm_len)
    }

    /// Convenience whole-buffer decode. Streaming users should retain state
    /// and call [`decode_into`](Self::decode_into).
    ///
    /// # Errors
    ///
    /// Propagates shape/configuration errors from `state` and `decode_into`.
    pub fn decode_all(&self, features: &[f32]) -> Result<Vec<f32>> {
        let feature_dim = self.config.input_dim;
        if features.is_empty() || features.len() % feature_dim != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "nanocodec causal HiFi-GAN decode: feature length {} is not a positive multiple of input_dim {feature_dim}",
                features.len()
            )));
        }
        let frames = features.len() / feature_dim;
        let mut state = self.state(frames)?;
        let pcm_len = frames.checked_mul(self.config.frame_hop).ok_or_else(|| {
            VokraError::InvalidArgument(
                "nanocodec causal HiFi-GAN decode: PCM length overflow".into(),
            )
        })?;
        let mut pcm = vec![0.0; pcm_len];
        self.decode_into(&mut state, features, &mut pcm)?;
        Ok(pcm)
    }

    fn validate_state(&self, state: &CausalHifiGanState) -> Result<()> {
        if state.geometry_signature != self.geometry_signature {
            return Err(VokraError::InvalidArgument(
                "nanocodec causal HiFi-GAN: state belongs to different decoder geometry".into(),
            ));
        }
        Ok(())
    }
}

fn load_error(message: String) -> VokraError {
    VokraError::ModelLoad(format!("nanocodec causal HiFi-GAN GGUF: {message}"))
}

fn require_metadata_usize(file: &GgufFile, key: &str) -> Result<usize> {
    match file.get(key) {
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| load_error(format!("metadata `{key}` is not a usize-range integer"))),
        None => Err(load_error(format!("required metadata `{key}` missing"))),
    }
}

fn require_metadata_usize_array(file: &GgufFile, key: &str) -> Result<Vec<usize>> {
    let values = match file.get(key) {
        Some(GgufMetadataValue::Array(array)) => &array.values,
        Some(_) => return Err(load_error(format!("metadata `{key}` is not an array"))),
        None => return Err(load_error(format!("required metadata `{key}` missing"))),
    };
    if values.is_empty() {
        return Err(load_error(format!("metadata array `{key}` is empty")));
    }
    values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|&value| value > 0)
                .ok_or_else(|| {
                    load_error(format!(
                        "metadata array `{key}` contains a non-positive or out-of-range integer"
                    ))
                })
        })
        .collect()
}

fn require_metadata_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::String(actual)) if actual == expected => Ok(()),
        Some(GgufMetadataValue::String(actual)) => Err(load_error(format!(
            "metadata `{key}` = `{actual}`, expected `{expected}`"
        ))),
        Some(_) => Err(load_error(format!("metadata `{key}` is not a string"))),
        None => Err(load_error(format!("required metadata `{key}` missing"))),
    }
}

fn require_metadata_nonempty_string(file: &GgufFile, key: &str) -> Result<String> {
    match file.get(key) {
        Some(GgufMetadataValue::String(actual)) if !actual.trim().is_empty() => {
            Ok(actual.to_owned())
        }
        Some(GgufMetadataValue::String(_)) => Err(load_error(format!("metadata `{key}` is empty"))),
        Some(_) => Err(load_error(format!("metadata `{key}` is not a string"))),
        None => Err(load_error(format!("required metadata `{key}` missing"))),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_audited_profile_metadata(
    source_model_id: &str,
    source_revision: &str,
    checkpoint_sha256: &str,
    sample_rate: usize,
    base_channels: usize,
    n_codebooks: usize,
    levels_per_group: &[usize],
    input_dim: usize,
    frame_hop: usize,
    upsample_rates: &[usize],
) -> Result<()> {
    if checkpoint_sha256.len() != 64
        || !checkpoint_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(load_error(
            "vokra.provenance.checkpoint_sha256 must be 64 lowercase hexadecimal characters"
                .to_owned(),
        ));
    }

    #[cfg(test)]
    if source_model_id == "nvidia/nemo-nano-codec-22khz-test-fixture" {
        if source_revision != REVISION_06 {
            return Err(load_error("test fixture revision mismatch".to_owned()));
        }
        return Ok(());
    }

    let (
        expected_revision,
        expected_groups,
        expected_levels,
        expected_dim,
        expected_hop,
        expected_rates,
    ) = match source_model_id {
        PROFILE_06 => (
            REVISION_06,
            4,
            &[9, 8, 8, 7][..],
            16,
            1764,
            &[7, 7, 6, 3, 2][..],
        ),
        PROFILE_178 => (
            REVISION_178,
            13,
            &[8, 7, 6, 6][..],
            52,
            1764,
            &[7, 7, 6, 3, 2][..],
        ),
        PROFILE_189 => (
            REVISION_189,
            8,
            &[8, 7, 6, 6][..],
            32,
            1024,
            &[8, 8, 4, 2, 2][..],
        ),
        _ => {
            return Err(load_error(format!(
                "vokra.provenance.upstream_hf `{source_model_id}` is not an audited NanoCodec profile"
            )));
        }
    };
    if source_revision != expected_revision {
        return Err(load_error(format!(
            "source revision `{source_revision}` does not match audited `{expected_revision}` for `{source_model_id}`"
        )));
    }
    if sample_rate != 22_050
        || base_channels != 864
        || n_codebooks != expected_groups
        || levels_per_group != expected_levels
        || input_dim != expected_dim
        || frame_hop != expected_hop
        || upsample_rates != expected_rates
    {
        return Err(load_error(format!(
            "GGUF geometry does not match audited profile `{source_model_id}`"
        )));
    }
    Ok(())
}

fn require_metadata_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::Bool(actual)) if *actual == expected => Ok(()),
        Some(GgufMetadataValue::Bool(actual)) => Err(load_error(format!(
            "metadata `{key}` = `{actual}`, expected `{expected}`"
        ))),
        Some(_) => Err(load_error(format!("metadata `{key}` is not a bool"))),
        None => Err(load_error(format!("required metadata `{key}` missing"))),
    }
}

fn load_f32_tensor(file: &GgufFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| load_error(format!("required tensor `{name}` missing")))?;
    if info.dtype != GgmlType::F32 {
        return Err(load_error(format!(
            "tensor `{name}` must be F32, got {:?}",
            info.dtype
        )));
    }
    let expected_shape = shape.iter().map(|&dim| dim as u64).collect::<Vec<_>>();
    if info.dimensions != expected_shape {
        return Err(load_error(format!(
            "tensor `{name}` shape {:?} != {expected_shape:?}",
            info.dimensions
        )));
    }
    file.tensor_f32(name)
        .map_err(|error| load_error(format!("cannot decode tensor `{name}`: {error}")))
}

fn load_conv1d(
    file: &GgufFile,
    prefix: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
) -> Result<CausalHifiGanConv1dWeights> {
    Ok(CausalHifiGanConv1dWeights {
        weight: load_f32_tensor(
            file,
            &format!("{prefix}.weight"),
            &[out_channels, in_channels, kernel],
        )?,
        bias: load_f32_tensor(file, &format!("{prefix}.bias"), &[out_channels])?,
    })
}

fn load_half_snake(
    file: &GgufFile,
    prefix: &str,
    channels: usize,
) -> Result<CausalHifiGanHalfSnakeWeights> {
    let snake_channels = channels / 2;
    Ok(CausalHifiGanHalfSnakeWeights {
        alpha: load_f32_tensor(file, &format!("{prefix}.alpha"), &[snake_channels])?,
        alpha_inv: load_f32_tensor(file, &format!("{prefix}.alpha_inv"), &[snake_channels])?,
    })
}

fn bind_conv(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    dilation: usize,
    weights: CausalHifiGanConv1dWeights,
    context: &str,
) -> Result<CausalConv1d> {
    let expected = checked_volume(out_channels, in_channels, kernel, context)?;
    if weights.weight.len() != expected || weights.bias.len() != out_channels {
        return Err(VokraError::InvalidArgument(format!(
            "nanocodec causal HiFi-GAN {context}: conv weight/bias lengths ({}/{}) != expected ({expected}/{out_channels})",
            weights.weight.len(),
            weights.bias.len()
        )));
    }
    CausalConv1d::new(
        in_channels,
        out_channels,
        kernel,
        1,
        dilation,
        &weights.weight,
        Some(weights.bias),
    )
}

fn bind_conv_transpose(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    weights: CausalHifiGanConvTranspose1dWeights,
    context: &str,
) -> Result<CausalConvTranspose1d> {
    let expected = checked_volume(in_channels, out_channels, kernel, context)?;
    if weights.weight.len() != expected || weights.bias.len() != out_channels {
        return Err(VokraError::InvalidArgument(format!(
            "nanocodec causal HiFi-GAN {context}: conv-transpose weight/bias lengths ({}/{}) != expected ({expected}/{out_channels})",
            weights.weight.len(),
            weights.bias.len()
        )));
    }
    CausalConvTranspose1d::new(
        in_channels,
        out_channels,
        kernel,
        stride,
        weights.weight,
        Some(weights.bias),
    )
}

fn checked_area(a: usize, b: usize, context: &str) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "nanocodec causal HiFi-GAN {context}: shape product overflow"
        ))
    })
}

fn checked_volume(a: usize, b: usize, c: usize, context: &str) -> Result<usize> {
    checked_area(a, b, context)?.checked_mul(c).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "nanocodec causal HiFi-GAN {context}: shape product overflow"
        ))
    })
}

fn geometry_signature(config: &CausalHifiGanConfig) -> u64 {
    // Deterministic FNV-1a over geometry only. This is a misuse guard, not a
    // cryptographic identifier; state carries no model-weight data.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |value: usize| {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    feed(config.input_dim);
    feed(config.base_channels);
    feed(config.frame_hop);
    feed(config.input_kernel_size);
    feed(config.output_kernel_size);
    feed(config.upsample_rates.len());
    for &value in &config.upsample_rates {
        feed(value);
    }
    feed(config.resblock_kernel_sizes.len());
    for &value in &config.resblock_kernel_sizes {
        feed(value);
    }
    feed(config.resblock_dilations.len());
    for &value in &config.resblock_dilations {
        feed(value);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{
        GgmlType, GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType,
    };

    fn values(n: usize, scale: f32, phase: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (((i as f32 + phase) * 0.173).sin()) * scale)
            .collect()
    }

    fn conv(out_ch: usize, in_ch: usize, kernel: usize, phase: f32) -> CausalHifiGanConv1dWeights {
        CausalHifiGanConv1dWeights {
            weight: values(out_ch * in_ch * kernel, 0.08, phase),
            bias: values(out_ch, 0.01, phase + 3.0),
        }
    }

    fn half_snake(channels: usize, phase: f32) -> CausalHifiGanHalfSnakeWeights {
        let alpha = values(channels / 2, 0.2, phase)
            .into_iter()
            .map(|v| v + 1.0)
            .collect::<Vec<_>>();
        let alpha_inv = alpha.iter().map(|a| 1.0 / (a + 1.0e-9)).collect();
        CausalHifiGanHalfSnakeWeights { alpha, alpha_inv }
    }

    fn fixture_parts(frame_hop: usize) -> (CausalHifiGanConfig, CausalHifiGanWeights) {
        // Two tiny grouped stages. The checkpoint-derived hop is deliberately
        // not either public NanoCodec variant's 1024 / 1764 value.
        let config = CausalHifiGanConfig {
            input_dim: 3,
            base_channels: 8,
            frame_hop,
            upsample_rates: vec![2, 3],
            input_kernel_size: 3,
            output_kernel_size: 3,
            resblock_kernel_sizes: vec![3, 5],
            resblock_dilations: vec![1, 2],
        };
        let mut in_ch = config.base_channels;
        let mut stages = Vec::new();
        for (stage_index, &rate) in config.upsample_rates.iter().enumerate() {
            let out_ch = in_ch / 2;
            let kernel = 2 * rate;
            let mut residual_branches = Vec::new();
            for (kernel_index, &res_kernel) in config.resblock_kernel_sizes.iter().enumerate() {
                let mut blocks = Vec::new();
                for (dilation_index, &dilation) in config.resblock_dilations.iter().enumerate() {
                    let phase = 20.0 * stage_index as f32
                        + 5.0 * kernel_index as f32
                        + dilation_index as f32;
                    blocks.push(CausalHifiGanResidualBlockWeights {
                        input_activation: half_snake(out_ch, phase),
                        input_conv: conv(out_ch, out_ch, res_kernel, phase + 1.0),
                        skip_activation: half_snake(out_ch, phase + 2.0),
                        skip_conv: conv(out_ch, out_ch, res_kernel, phase + 3.0),
                        dilation,
                    });
                }
                residual_branches.push(blocks);
            }
            stages.push(CausalHifiGanStageWeights {
                activation: half_snake(in_ch, 40.0 + stage_index as f32),
                // NeMo-Speech.cpp expands the checkpoint's grouped
                // ConvTranspose1d tensor to dense [in, out, kernel] at
                // conversion time; zeros preserve the groups exactly.
                upsample: CausalHifiGanConvTranspose1dWeights {
                    weight: grouped_dense_weight(in_ch, out_ch, kernel, stage_index as f32),
                    bias: values(out_ch, 0.01, 50.0 + stage_index as f32),
                },
                residual_branches,
            });
            in_ch = out_ch;
        }
        let weights = CausalHifiGanWeights {
            pre_conv: conv(
                config.base_channels,
                config.input_dim,
                config.input_kernel_size,
                70.0,
            ),
            stages,
            post_activation: half_snake(in_ch, 80.0),
            post_conv: conv(1, in_ch, config.output_kernel_size, 90.0),
        };
        (config, weights)
    }

    fn fixture_with_hop(frame_hop: usize) -> CausalHifiGan {
        let (config, weights) = fixture_parts(frame_hop);
        CausalHifiGan::new(config, weights).expect("tiny causal HiFi-GAN")
    }

    fn grouped_dense_weight(in_ch: usize, out_ch: usize, kernel: usize, phase: f32) -> Vec<f32> {
        assert_eq!(in_ch, 2 * out_ch);
        let mut dense = vec![0.0; in_ch * out_ch * kernel];
        for input in 0..in_ch {
            let output = input / 2;
            for tap in 0..kernel {
                dense[(input * out_ch + output) * kernel + tap] =
                    (((input + tap) as f32 + phase) * 0.119).sin() * 0.07;
            }
        }
        dense
    }

    fn features(frames: usize, dim: usize) -> Vec<f32> {
        values(frames * dim, 0.4, 101.0)
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn add_tensor(builder: &mut GgufBuilder, name: &str, shape: &[usize], values: &[f32]) {
        builder
            .add_tensor(
                name,
                GgmlType::F32,
                shape.iter().map(|&dim| dim as u64).collect(),
                f32_bytes(values),
            )
            .expect("add fixture tensor");
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

    fn fixture_gguf(
        config: &CausalHifiGanConfig,
        weights: &CausalHifiGanWeights,
        arch: &str,
        corrupt_pre_conv_shape: bool,
        source_model_id: &str,
    ) -> GgufFile {
        let mut builder = GgufBuilder::new();
        builder.add_string("vokra.model.arch", arch);
        builder.add_u32("vokra.nanocodec.n_codebooks", 1);
        add_u32_array(&mut builder, "vokra.nanocodec.levels_per_group", &[2, 2, 2]);
        builder.add_u32("vokra.nanocodec.embed_dim", config.input_dim as u32);
        builder.add_u32("vokra.nanocodec.sample_rate", 22_050);
        builder.add_u32("vokra.nanocodec.frame_hop", config.frame_hop as u32);
        builder.add_u32("vokra.nanocodec.generator_hop", config.frame_hop as u32);
        builder.add_u32("vokra.nanocodec.base_channels", config.base_channels as u32);
        add_u32_array(
            &mut builder,
            "vokra.nanocodec.upsample_rates",
            &config.upsample_rates,
        );
        builder.add_u32(
            "vokra.nanocodec.input_kernel_size",
            config.input_kernel_size as u32,
        );
        builder.add_u32(
            "vokra.nanocodec.output_kernel_size",
            config.output_kernel_size as u32,
        );
        add_u32_array(
            &mut builder,
            "vokra.nanocodec.resblock_kernel_sizes",
            &config.resblock_kernel_sizes,
        );
        add_u32_array(
            &mut builder,
            "vokra.nanocodec.resblock_dilations",
            &config.resblock_dilations,
        );
        builder.add_string("vokra.nanocodec.activation", "HalfSnake");
        builder.add_string("vokra.nanocodec.output_activation", "ClampActivation");
        builder.add_string("vokra.nanocodec.pad_mode", "zeros");
        builder.add_string("vokra.nanocodec.nemo_speech_commit", NEMO_SPEECH_COMMIT);
        builder.add_string("vokra.nanocodec.nemo_source_url", NEMO_SOURCE_URL);
        builder.add_bool("vokra.nanocodec.grouped_upsample_expanded", true);
        builder.add_string("vokra.provenance.upstream_hf", source_model_id);
        builder.add_string(
            "vokra.provenance.upstream_revision",
            "5c8e22ed763c14d81337fbe6ca74062f3d10f7e5",
        );
        builder.add_string(
            "vokra.provenance.checkpoint_sha256",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        );

        if corrupt_pre_conv_shape {
            add_tensor(
                &mut builder,
                "nanocodec.pre_conv.weight",
                &[1, 1, 1],
                &[0.0],
            );
        } else {
            add_tensor(
                &mut builder,
                "nanocodec.pre_conv.weight",
                &[
                    config.base_channels,
                    config.input_dim,
                    config.input_kernel_size,
                ],
                &weights.pre_conv.weight,
            );
        }
        add_tensor(
            &mut builder,
            "nanocodec.pre_conv.bias",
            &[config.base_channels],
            &weights.pre_conv.bias,
        );
        let mut in_channels = config.base_channels;
        for (stage_index, (stage, &rate)) in weights
            .stages
            .iter()
            .zip(config.upsample_rates.iter())
            .enumerate()
        {
            let out_channels = in_channels / 2;
            let prefix = format!("nanocodec.stage.{stage_index}");
            add_tensor(
                &mut builder,
                &format!("{prefix}.activation.alpha"),
                &[in_channels / 2],
                &stage.activation.alpha,
            );
            add_tensor(
                &mut builder,
                &format!("{prefix}.activation.alpha_inv"),
                &[in_channels / 2],
                &stage.activation.alpha_inv,
            );
            add_tensor(
                &mut builder,
                &format!("{prefix}.upsample.weight"),
                &[in_channels, out_channels, 2 * rate],
                &stage.upsample.weight,
            );
            add_tensor(
                &mut builder,
                &format!("{prefix}.upsample.bias"),
                &[out_channels],
                &stage.upsample.bias,
            );
            for (branch_index, branch) in stage.residual_branches.iter().enumerate() {
                let kernel = config.resblock_kernel_sizes[branch_index];
                for (block_index, block) in branch.iter().enumerate() {
                    let block_prefix =
                        format!("{prefix}.branch.{branch_index}.block.{block_index}");
                    for (activation_name, activation) in [
                        ("input_activation", &block.input_activation),
                        ("skip_activation", &block.skip_activation),
                    ] {
                        add_tensor(
                            &mut builder,
                            &format!("{block_prefix}.{activation_name}.alpha"),
                            &[out_channels / 2],
                            &activation.alpha,
                        );
                        add_tensor(
                            &mut builder,
                            &format!("{block_prefix}.{activation_name}.alpha_inv"),
                            &[out_channels / 2],
                            &activation.alpha_inv,
                        );
                    }
                    for (conv_name, conv) in [
                        ("input_conv", &block.input_conv),
                        ("skip_conv", &block.skip_conv),
                    ] {
                        add_tensor(
                            &mut builder,
                            &format!("{block_prefix}.{conv_name}.weight"),
                            &[out_channels, out_channels, kernel],
                            &conv.weight,
                        );
                        add_tensor(
                            &mut builder,
                            &format!("{block_prefix}.{conv_name}.bias"),
                            &[out_channels],
                            &conv.bias,
                        );
                    }
                }
            }
            in_channels = out_channels;
        }
        add_tensor(
            &mut builder,
            "nanocodec.post_activation.alpha",
            &[in_channels / 2],
            &weights.post_activation.alpha,
        );
        add_tensor(
            &mut builder,
            "nanocodec.post_activation.alpha_inv",
            &[in_channels / 2],
            &weights.post_activation.alpha_inv,
        );
        add_tensor(
            &mut builder,
            "nanocodec.post_conv.weight",
            &[1, in_channels, config.output_kernel_size],
            &weights.post_conv.weight,
        );
        add_tensor(
            &mut builder,
            "nanocodec.post_conv.bias",
            &[1],
            &weights.post_conv.bias,
        );
        GgufFile::parse(builder.to_bytes().expect("serialize fixture GGUF"))
            .expect("parse fixture GGUF")
    }

    #[test]
    fn converter_gguf_contract_binds_and_decodes_identically() {
        let (config, weights) = fixture_parts(6);
        let direct = CausalHifiGan::new(config.clone(), weights.clone()).unwrap();
        let file = fixture_gguf(
            &config,
            &weights,
            "nanocodec",
            false,
            "nvidia/nemo-nano-codec-22khz-test-fixture",
        );
        let rebound = CausalHifiGan::from_gguf(&file).expect("bind converter GGUF");
        let input = features(4, config.input_dim);
        assert_eq!(
            direct.decode_all(&input).unwrap(),
            rebound.decode_all(&input).unwrap()
        );
    }

    #[test]
    fn gguf_binder_rejects_foreign_architecture() {
        let (config, weights) = fixture_parts(6);
        let file = fixture_gguf(
            &config,
            &weights,
            "mimi",
            false,
            "nvidia/nemo-nano-codec-22khz-test-fixture",
        );
        let error = CausalHifiGan::from_gguf(&file).unwrap_err();
        assert!(error.to_string().contains("vokra.model.arch"), "{error}");
    }

    #[test]
    fn gguf_binder_rejects_missing_metadata_and_tensor_shape() {
        let mut incomplete = GgufBuilder::new();
        incomplete.add_string("vokra.model.arch", "nanocodec");
        let incomplete = GgufFile::parse(incomplete.to_bytes().unwrap()).unwrap();
        let error = CausalHifiGan::from_gguf(&incomplete).unwrap_err();
        assert!(
            error.to_string().contains("vokra.nanocodec.activation"),
            "{error}"
        );

        let (config, weights) = fixture_parts(6);
        let wrong_shape = fixture_gguf(
            &config,
            &weights,
            "nanocodec",
            true,
            "nvidia/nemo-nano-codec-22khz-test-fixture",
        );
        let error = CausalHifiGan::from_gguf(&wrong_shape).unwrap_err();
        assert!(error.to_string().contains("pre_conv.weight"), "{error}");
        assert!(error.to_string().contains("shape"), "{error}");
    }

    #[test]
    fn gguf_binder_rejects_mislabeled_official_profile() {
        let (config, weights) = fixture_parts(6);
        let mislabeled = fixture_gguf(
            &config,
            &weights,
            "nanocodec",
            false,
            "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps",
        );
        let error = CausalHifiGan::from_gguf(&mislabeled).unwrap_err();
        assert!(error.to_string().contains("audited profile"), "{error}");
    }

    #[test]
    fn whole_buffer_is_bit_identical_to_frame_streaming() {
        let decoder = fixture_with_hop(6);
        let input = features(5, decoder.expected_feature_dim());
        let whole = decoder.decode_all(&input).unwrap();

        let mut state = decoder.state(1).unwrap();
        let mut streamed = vec![0.0; whole.len()];
        for frame in 0..5 {
            let feature_start = frame * decoder.expected_feature_dim();
            let pcm_start = frame * decoder.frame_hop();
            let written = decoder
                .decode_into(
                    &mut state,
                    &input[feature_start..feature_start + decoder.expected_feature_dim()],
                    &mut streamed[pcm_start..pcm_start + decoder.frame_hop()],
                )
                .unwrap();
            assert_eq!(written, decoder.frame_hop());
        }
        assert_eq!(whole, streamed);
    }

    #[test]
    fn future_frames_do_not_change_emitted_samples() {
        let decoder = fixture_with_hop(6);
        let input = features(6, decoder.expected_feature_dim());
        let prefix_frames = 2;
        let prefix = decoder
            .decode_all(&input[..prefix_frames * decoder.expected_feature_dim()])
            .unwrap();
        let with_future = decoder.decode_all(&input).unwrap();
        assert_eq!(prefix, with_future[..prefix_frames * decoder.frame_hop()]);
    }

    #[test]
    fn reset_replays_the_stream_exactly() {
        let decoder = fixture_with_hop(6);
        let input = features(2, decoder.expected_feature_dim());
        let mut state = decoder.state(2).unwrap();
        let mut first = vec![0.0; 2 * decoder.frame_hop()];
        decoder.decode_into(&mut state, &input, &mut first).unwrap();
        decoder.reset(&mut state).unwrap();
        let mut second = vec![0.0; 2 * decoder.frame_hop()];
        decoder
            .decode_into(&mut state, &input, &mut second)
            .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn frame_hop_is_checkpoint_driven_and_cross_checked_with_generator() {
        let decoder = fixture_with_hop(6);
        assert_eq!(decoder.frame_hop(), 6);
        assert_eq!(decoder.decode_all(&features(2, 3)).unwrap().len(), 12);

        let public_12_5_fps = CausalHifiGanConfig {
            input_dim: 16,
            base_channels: 864,
            frame_hop: 1764,
            upsample_rates: vec![7, 7, 6, 3, 2],
            input_kernel_size: 3,
            output_kernel_size: 3,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilations: vec![1, 3, 5],
        };
        public_12_5_fps
            .validate()
            .expect("checkpoint-derived 1764 hop is accepted without a runtime default");

        let (mut config, weights) = fixture_parts(6);
        config.frame_hop = 7;
        let err = CausalHifiGan::new(config, weights).unwrap_err();
        assert!(err.to_string().contains("frame_hop"), "{err}");
    }

    #[test]
    fn published_21_5_fps_geometry_is_consistent_and_mismatch_fails_closed() {
        let config = CausalHifiGanConfig {
            input_dim: 32,
            base_channels: 864,
            frame_hop: 1024,
            upsample_rates: vec![8, 8, 4, 2, 2],
            input_kernel_size: 7,
            output_kernel_size: 3,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilations: vec![1, 3, 5],
        };
        config
            .validate()
            .expect("official 1.89 kbps checkpoint geometry must be accepted");
        assert_eq!(config.upsample_rates.iter().product::<usize>(), 1024);
        assert_eq!(config.frame_hop, 1024);

        let mut inconsistent = config;
        inconsistent.frame_hop = 2048;
        let error = inconsistent.validate().unwrap_err();
        assert!(error.to_string().contains("refusing to drop"), "{error}");
    }

    #[test]
    fn invalid_shapes_and_capacity_fail_loudly() {
        let decoder = fixture_with_hop(6);
        assert!(decoder.state(0).is_err());
        let mut state = decoder.state(1).unwrap();
        let mut pcm = vec![0.0; decoder.frame_hop()];
        assert!(decoder.decode_into(&mut state, &[], &mut pcm).is_err());
        assert!(
            decoder
                .decode_into(
                    &mut state,
                    &features(2, decoder.expected_feature_dim()),
                    &mut pcm,
                )
                .is_err()
        );
        let short_pcm_len = pcm.len() - 1;
        assert!(
            decoder
                .decode_into(
                    &mut state,
                    &features(1, decoder.expected_feature_dim()),
                    &mut pcm[..short_pcm_len],
                )
                .is_err()
        );

        let (config, mut weights) = fixture_parts(6);
        weights.post_activation.alpha[0] = -HALF_SNAKE_EPSILON;
        weights.post_activation.alpha_inv[0] = 0.0;
        let error = CausalHifiGan::new(config, weights).unwrap_err();
        assert!(error.to_string().contains("alpha_inv"), "{error}");
    }
}
