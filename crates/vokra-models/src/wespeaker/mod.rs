//! Native WeSpeaker ResNet34-LM speaker embedding.
//!
//! The implementation follows the pinned upstream ResNet34 topology exactly:
//! 80-bin Hamming-window Kaldi fbank with utterance CMN, a `[3, 4, 6, 3]`
//! basic-block ResNet, temporal mean plus Bessel-corrected standard-deviation
//! pooling, and the 256-dimensional `seg_1` projection. The pyannote
//! diarization path additionally supports its frame mask through the pinned
//! weighted `StatsPool` contract. Every learned Conv2D and the final
//! projection are lowered to [`Compute::gemm_f32`], so a Metal selection is
//! observable and can never silently fall back to CPU.

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{BackendKind, LicenseClass, Result, SpeakerEngine, VokraError};
use vokra_ops::{KaldiFbankOpts, KaldiFbankWindow, kaldi_fbank_with_window};

use crate::compute::{Compute, HotOp};

/// `vokra.model.arch` accepted by this runtime.
pub const ARCH: &str = "wespeaker";
/// Canonical model identity shared by the two public layouts.
pub const NAME: &str = "wespeaker-voxceleb-resnet34-lm";
/// Model task category.
pub const CATEGORY: &str = "speaker";
/// Audited upstream Hugging Face repository.
pub const UPSTREAM_HF: &str = "Wespeaker/wespeaker-voxceleb-resnet34-LM";
/// Pinned upstream Hugging Face checkpoint revision.
pub const UPSTREAM_REVISION: &str = "f0c48c298fd835726c27956a5d617bad7115627e";
/// Pinned WeSpeaker source revision used by the native implementation.
pub const SOURCE_REVISION: &str = "45941e7cba2c3ea99e232d02bedf617fc71b0dad";
/// Output speaker-embedding width.
pub const EMBED_DIM: usize = 256;

const INPUT_DIM: usize = 80;
const STAGE_BLOCKS: [usize; 4] = [3, 4, 6, 3];
const STAGE_CHANNELS: [usize; 4] = [32, 64, 128, 256];
const FINAL_FREQ: usize = INPUT_DIM / 8;
const POOL_INPUT_DIM: usize = STAGE_CHANNELS[3] * FINAL_FREQ;
const STATS_DIM: usize = POOL_INPUT_DIM * 2;
const BN_EPS: f32 = 1.0e-5;
const STATS_EPS: f32 = 1.0e-7;
const WEIGHTED_STATS_EPS: f32 = 1.0e-8;
const PREFIXED_TENSOR_COUNT: usize = 182;
const BARE_COMBINED_TENSOR_COUNT: usize = 219;
const WESPEAKER_HOT_OPS: &[HotOp] = &[HotOp::Gemm];

const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_SOURCE_REVISION: &str = "vokra.wespeaker.source_revision";
const KEY_SAMPLE_RATE: &str = "vokra.wespeaker.sample_rate";
const KEY_N_MELS: &str = "vokra.wespeaker.n_mels";
const KEY_FRAME_LENGTH: &str = "vokra.wespeaker.frame_length";
const KEY_FRAME_SHIFT: &str = "vokra.wespeaker.frame_shift";
const KEY_EMBED_DIM: &str = "vokra.wespeaker.embed_dim";
const KEY_STAGE_COUNT: &str = "vokra.wespeaker.stage_count";
const KEY_BN_EPS: &str = "vokra.wespeaker.bn_eps";
const KEY_STATS_EPS: &str = "vokra.wespeaker.stats_eps";
const KEY_FRONTEND: &str = "vokra.wespeaker.frontend";
const KEY_BLOCKS: &str = "vokra.wespeaker.blocks";
const KEY_CHANNELS: &str = "vokra.wespeaker.channels";
const KEY_POOLING: &str = "vokra.wespeaker.pooling";
const KEY_LAYOUT: &str = "vokra.wespeaker.artifact_layout";
const CONTRACT_KEYS: [&str; 14] = [
    KEY_SAMPLE_RATE,
    KEY_N_MELS,
    KEY_FRAME_LENGTH,
    KEY_FRAME_SHIFT,
    KEY_EMBED_DIM,
    KEY_STAGE_COUNT,
    KEY_BN_EPS,
    KEY_STATS_EPS,
    KEY_FRONTEND,
    KEY_BLOCKS,
    KEY_CHANNELS,
    KEY_POOLING,
    KEY_LAYOUT,
    KEY_SOURCE_REVISION,
];

/// Exact public WeSpeaker tensor-layout variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeSpeakerArtifactLayout {
    /// `resnet.*` embedding-only tensors used by the pyannote artifact.
    PyannotePrefixed,
    /// Bare backbone plus training counters and the unused LM classifier.
    OfficialCombinedBare,
}

impl WeSpeakerArtifactLayout {
    const fn stem(self) -> &'static str {
        match self {
            Self::PyannotePrefixed => "resnet.",
            Self::OfficialCombinedBare => "",
        }
    }

    const fn tensor_count(self) -> usize {
        match self {
            Self::PyannotePrefixed => PREFIXED_TENSOR_COUNT,
            Self::OfficialCombinedBare => BARE_COMBINED_TENSOR_COUNT,
        }
    }

    const fn contract_name(self) -> &'static str {
        match self {
            Self::PyannotePrefixed => "pyannote-prefixed-182-v1",
            Self::OfficialCombinedBare => "official-combined-bare-219-v1",
        }
    }
}

#[derive(Debug)]
struct FoldedBatchNorm {
    scale: Vec<f32>,
    shift: Vec<f32>,
}

impl FoldedBatchNorm {
    fn bind(file: &GgufFile, prefix: &str, channels: usize) -> Result<Self> {
        let gamma = tensor(file, &format!("{prefix}.weight"), &[channels])?;
        let beta = tensor(file, &format!("{prefix}.bias"), &[channels])?;
        let mean = tensor(file, &format!("{prefix}.running_mean"), &[channels])?;
        let variance = tensor(file, &format!("{prefix}.running_var"), &[channels])?;
        let mut scale = vec![0.0; channels];
        let mut shift = vec![0.0; channels];
        for channel in 0..channels {
            scale[channel] = gamma[channel] / (variance[channel] + BN_EPS).sqrt();
            shift[channel] = beta[channel] - mean[channel] * scale[channel];
        }
        Ok(Self { scale, shift })
    }

    fn apply(&self, values: &mut [f32], spatial: usize) {
        debug_assert_eq!(values.len(), self.scale.len() * spatial);
        for channel in 0..self.scale.len() {
            let scale = self.scale[channel];
            let shift = self.shift[channel];
            for value in &mut values[channel * spatial..(channel + 1) * spatial] {
                *value = *value * scale + shift;
            }
        }
    }
}

#[derive(Debug)]
struct Conv2d {
    weight: Vec<f32>,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
}

impl Conv2d {
    fn bind(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: tensor(
                file,
                &format!("{prefix}.weight"),
                &[output_channels, input_channels, kernel, kernel],
            )?,
            input_channels,
            output_channels,
            kernel,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        height: usize,
        width: usize,
        stride: usize,
        padding: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize, usize)> {
        if input.len() != self.input_channels * height * width {
            return Err(VokraError::InvalidArgument(format!(
                "wespeaker: Conv2D input has {} values, expected {} x {height} x {width}",
                input.len(),
                self.input_channels
            )));
        }
        if height + 2 * padding < self.kernel || width + 2 * padding < self.kernel {
            return Err(VokraError::InvalidArgument(format!(
                "wespeaker: Conv2D input {height}x{width} is too small for kernel {} and padding {padding}",
                self.kernel
            )));
        }
        let out_height = (height + 2 * padding - self.kernel) / stride + 1;
        let out_width = (width + 2 * padding - self.kernel) / stride + 1;
        let spatial = out_height * out_width;
        let patch = self.input_channels * self.kernel * self.kernel;
        let mut output = vec![0.0; self.output_channels * spatial];

        crate::tls_scratch::with_col_scratch(patch * spatial, |columns| {
            for input_channel in 0..self.input_channels {
                let plane = input_channel * height * width;
                for kernel_y in 0..self.kernel {
                    for kernel_x in 0..self.kernel {
                        let row = ((input_channel * self.kernel + kernel_y) * self.kernel
                            + kernel_x)
                            * spatial;
                        for out_y in 0..out_height {
                            let destination = &mut columns
                                [row + out_y * out_width..row + (out_y + 1) * out_width];
                            let input_y = (out_y * stride + kernel_y) as isize - padding as isize;
                            for (out_x, value) in destination.iter_mut().enumerate() {
                                let input_x =
                                    (out_x * stride + kernel_x) as isize - padding as isize;
                                *value = if input_y >= 0
                                    && input_y < height as isize
                                    && input_x >= 0
                                    && input_x < width as isize
                                {
                                    input[plane + input_y as usize * width + input_x as usize]
                                } else {
                                    0.0
                                };
                            }
                        }
                    }
                }
            }
            compute.gemm_f32(
                self.output_channels,
                spatial,
                patch,
                &self.weight,
                columns,
                None,
                &mut output,
            )
        })?;
        Ok((output, out_height, out_width))
    }
}

#[derive(Debug)]
struct BasicBlock {
    conv1: Conv2d,
    bn1: FoldedBatchNorm,
    conv2: Conv2d,
    bn2: FoldedBatchNorm,
    shortcut: Option<(Conv2d, FoldedBatchNorm)>,
    stride: usize,
}

impl BasicBlock {
    fn bind(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        stride: usize,
    ) -> Result<Self> {
        let shortcut = if stride != 1 || input_channels != output_channels {
            Some((
                Conv2d::bind(
                    file,
                    &format!("{prefix}.shortcut.0"),
                    input_channels,
                    output_channels,
                    1,
                )?,
                FoldedBatchNorm::bind(file, &format!("{prefix}.shortcut.1"), output_channels)?,
            ))
        } else {
            None
        };
        Ok(Self {
            conv1: Conv2d::bind(
                file,
                &format!("{prefix}.conv1"),
                input_channels,
                output_channels,
                3,
            )?,
            bn1: FoldedBatchNorm::bind(file, &format!("{prefix}.bn1"), output_channels)?,
            conv2: Conv2d::bind(
                file,
                &format!("{prefix}.conv2"),
                output_channels,
                output_channels,
                3,
            )?,
            bn2: FoldedBatchNorm::bind(file, &format!("{prefix}.bn2"), output_channels)?,
            shortcut,
            stride,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        height: usize,
        width: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize, usize)> {
        let (mut hidden, out_height, out_width) =
            self.conv1
                .forward(input, height, width, self.stride, 1, compute)?;
        self.bn1.apply(&mut hidden, out_height * out_width);
        relu(&mut hidden);
        let (mut output, height2, width2) = self
            .conv2
            .forward(&hidden, out_height, out_width, 1, 1, compute)?;
        debug_assert_eq!((height2, width2), (out_height, out_width));
        self.bn2.apply(&mut output, out_height * out_width);

        if let Some((conv, norm)) = &self.shortcut {
            let (mut residual, residual_height, residual_width) =
                conv.forward(input, height, width, self.stride, 0, compute)?;
            if (residual_height, residual_width) != (out_height, out_width) {
                return Err(VokraError::InvalidArgument(
                    "wespeaker: residual and main Conv2D shapes disagree".into(),
                ));
            }
            norm.apply(&mut residual, out_height * out_width);
            for (value, skip) in output.iter_mut().zip(residual) {
                *value += skip;
            }
        } else {
            for (value, skip) in output.iter_mut().zip(input) {
                *value += skip;
            }
        }
        relu(&mut output);
        Ok((output, out_height, out_width))
    }
}

#[derive(Debug)]
struct WeSpeakerWeights {
    stem_conv: Conv2d,
    stem_norm: FoldedBatchNorm,
    stages: Vec<Vec<BasicBlock>>,
    projection_weight: Vec<f32>,
    projection_bias: Vec<f32>,
    layout: WeSpeakerArtifactLayout,
}

impl WeSpeakerWeights {
    fn bind(file: &GgufFile) -> Result<Self> {
        let layout = detect_layout(file)?;
        verify_manifest(file, layout)?;
        let stem = layout.stem();
        let mut input_channels = 32;
        let mut stages = Vec::with_capacity(STAGE_BLOCKS.len());
        for (stage_index, (&block_count, &output_channels)) in
            STAGE_BLOCKS.iter().zip(&STAGE_CHANNELS).enumerate()
        {
            let mut blocks = Vec::with_capacity(block_count);
            for block_index in 0..block_count {
                let stride = if stage_index > 0 && block_index == 0 {
                    2
                } else {
                    1
                };
                blocks.push(BasicBlock::bind(
                    file,
                    &format!("{stem}layer{}.{}", stage_index + 1, block_index),
                    input_channels,
                    output_channels,
                    stride,
                )?);
                input_channels = output_channels;
            }
            stages.push(blocks);
        }
        Ok(Self {
            stem_conv: Conv2d::bind(file, &format!("{stem}conv1"), 1, 32, 3)?,
            stem_norm: FoldedBatchNorm::bind(file, &format!("{stem}bn1"), 32)?,
            stages,
            projection_weight: tensor(
                file,
                &format!("{stem}seg_1.weight"),
                &[EMBED_DIM, STATS_DIM],
            )?,
            projection_bias: tensor(file, &format!("{stem}seg_1.bias"), &[EMBED_DIM])?,
            layout,
        })
    }

    fn embed_features(
        &self,
        features: &[f32],
        frames: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        self.embed_features_impl(features, frames, None, compute)
    }

    fn embed_features_masked(
        &self,
        features: &[f32],
        frames: usize,
        mask: &[f32],
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        self.embed_features_impl(features, frames, Some(mask), compute)
    }

    fn embed_features_impl(
        &self,
        features: &[f32],
        frames: usize,
        mask: Option<&[f32]>,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if features.len() != frames * INPUT_DIM {
            return Err(VokraError::InvalidArgument(format!(
                "wespeaker: feature buffer has {} values, expected {frames} x {INPUT_DIM}",
                features.len()
            )));
        }
        if frames < 9 {
            return Err(VokraError::InvalidArgument(format!(
                "wespeaker: {frames} feature frames are too short; need at least 9 so TSTP has two final frames"
            )));
        }
        let mut input = vec![0.0; INPUT_DIM * frames];
        for frame in 0..frames {
            for frequency in 0..INPUT_DIM {
                input[frequency * frames + frame] = features[frame * INPUT_DIM + frequency];
            }
        }

        let (mut hidden, mut height, mut width) = self
            .stem_conv
            .forward(&input, INPUT_DIM, frames, 1, 1, compute)?;
        self.stem_norm.apply(&mut hidden, height * width);
        relu(&mut hidden);
        for stage in &self.stages {
            for block in stage {
                (hidden, height, width) = block.forward(&hidden, height, width, compute)?;
            }
        }
        if height != FINAL_FREQ {
            return Err(VokraError::InvalidArgument(format!(
                "wespeaker: final frequency dimension is {height}, expected {FINAL_FREQ}"
            )));
        }
        let statistics = match mask {
            Some(mask) => weighted_temporal_statistics_pool(&hidden, POOL_INPUT_DIM, width, mask)?,
            None => temporal_statistics_pool(&hidden, POOL_INPUT_DIM, width)?,
        };
        let mut embedding = vec![0.0; EMBED_DIM];
        compute.gemm_f32(
            EMBED_DIM,
            1,
            STATS_DIM,
            &self.projection_weight,
            &statistics,
            None,
            &mut embedding,
        )?;
        for (value, bias) in embedding.iter_mut().zip(&self.projection_bias) {
            *value += bias;
        }
        Ok(embedding)
    }
}

/// Complete native WeSpeaker ResNet34-LM inference handle.
#[derive(Debug)]
pub struct WeSpeaker {
    weights: WeSpeakerWeights,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl WeSpeaker {
    /// Strictly binds one of the two exact public tensor manifests.
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
                "wespeaker: `{}` is missing or empty for the CC-BY-4.0 checkpoint",
                chunks::KEY_PROVENANCE_ATTRIBUTION
            )));
        }
        let weights = WeSpeakerWeights::bind(file)?;
        verify_optional_contract(file, weights.layout)?;
        Ok(Self {
            weights,
            weight_license: LicenseClass::AttributionRequired,
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and strictly binds a GGUF file.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Selects one backend for all learned convolutions and projection GEMMs.
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

    /// Returns the bound public tensor-layout variant.
    #[must_use]
    pub const fn artifact_layout(&self) -> WeSpeakerArtifactLayout {
        self.weights.layout
    }

    /// Returns the normalized provenance license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Runs the official PCM frontend and complete speaker network.
    pub fn embed_pcm(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let options = frontend_options();
        if sample_rate != options.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "wespeaker: expected {} Hz mono PCM, got {sample_rate} Hz; resample offline first",
                options.sample_rate
            )));
        }
        let (features, frames) = kaldi_fbank_with_window(pcm, &options, KaldiFbankWindow::Hamming)?;
        self.embed_features(&features, frames)
    }

    /// Runs the official PCM frontend and speaker network with one pyannote
    /// activity mask.
    ///
    /// `mask` is the binarized PyanNet activity for one local speaker. Its
    /// length may differ from both the fbank and final ResNet frame counts;
    /// the pinned upstream `StatsPool` nearest-neighbor interpolates it
    /// directly to the final time axis. Values must be finite and in
    /// `[0, 1]`. An all-zero mask is valid and yields zero pooled statistics
    /// before the learned projection bias, matching upstream behavior.
    pub fn embed_pcm_masked(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        mask: &[f32],
    ) -> Result<Vec<f32>> {
        let options = frontend_options();
        if sample_rate != options.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "wespeaker: expected {} Hz mono PCM, got {sample_rate} Hz; resample offline first",
                options.sample_rate
            )));
        }
        let (features, frames) = kaldi_fbank_with_window(pcm, &options, KaldiFbankWindow::Hamming)?;
        self.embed_features_masked(&features, frames, mask)
    }

    /// Runs only the learned network on row-major `[frames, 80]` features.
    pub fn embed_features(&self, features: &[f32], frames: usize) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, WESPEAKER_HOT_OPS)?;
        self.weights.embed_features(features, frames, &compute)
    }

    /// Runs the learned network on row-major `[frames, 80]` features and
    /// applies one pyannote local-speaker activity mask at `StatsPool`.
    ///
    /// The mask interpolation and weighted-statistics contract are identical
    /// to [`Self::embed_pcm_masked`].
    pub fn embed_features_masked(
        &self,
        features: &[f32],
        frames: usize,
        mask: &[f32],
    ) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, WESPEAKER_HOT_OPS)?;
        self.weights
            .embed_features_masked(features, frames, mask, &compute)
    }

    /// Computes the official Hamming Kaldi-fbank frontend for parity checks.
    pub fn frontend_features(&self, pcm: &[f32], sample_rate: u32) -> Result<(Vec<f32>, usize)> {
        let options = frontend_options();
        if sample_rate != options.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "wespeaker: expected {} Hz mono PCM, got {sample_rate} Hz",
                options.sample_rate
            )));
        }
        kaldi_fbank_with_window(pcm, &options, KaldiFbankWindow::Hamming)
    }
}

impl SpeakerEngine for WeSpeaker {
    fn embed(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        self.embed_pcm(pcm, sample_rate)
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }
}

fn frontend_options() -> KaldiFbankOpts {
    KaldiFbankOpts {
        sample_rate: 16_000,
        num_mel_bins: INPUT_DIM,
        frame_length: 400,
        frame_shift: 160,
        remove_dc_offset: true,
        preemph_coeff: 0.97,
        low_freq: 20.0,
        high_freq: 0.0,
        use_power: true,
        use_log: true,
        subtract_mean: true,
        round_to_power_of_two: true,
    }
}

fn relu(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

fn temporal_statistics_pool(input: &[f32], channels: usize, frames: usize) -> Result<Vec<f32>> {
    if input.len() != channels * frames || frames < 2 {
        return Err(VokraError::InvalidArgument(format!(
            "wespeaker: TSTP requires [channels={channels}, frames>=2], got {} values and {frames} frames",
            input.len()
        )));
    }
    let mut output = vec![0.0; channels * 2];
    for (channel, row) in input.chunks_exact(frames).enumerate() {
        let mean = row.iter().copied().sum::<f32>() / frames as f32;
        let variance = row
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / (frames - 1) as f32;
        output[channel] = mean;
        output[channels + channel] = (variance + STATS_EPS).sqrt();
    }
    Ok(output)
}

/// pyannote.audio 3.1.1 weighted `StatsPool` for one local-speaker mask.
///
/// Primary source:
/// `pyannote/audio/models/blocks/pooling.py` at revision
/// `6a972c0c4e95de04637d7221208736c64c8b972a`. The source interpolates an
/// arbitrary-length mask to the final ResNet time axis with PyTorch
/// `mode="nearest"`, then computes:
///
/// - `v1 = sum(w) + 1e-8`
/// - `mean = sum(x*w) / v1`
/// - `v2 = sum(w²)`
/// - `var = sum((x-mean)²*w) / (v1 - v2/v1 + 1e-8)`
fn weighted_temporal_statistics_pool(
    input: &[f32],
    channels: usize,
    frames: usize,
    mask: &[f32],
) -> Result<Vec<f32>> {
    if input.len() != channels * frames || frames < 2 {
        return Err(VokraError::InvalidArgument(format!(
            "wespeaker: weighted TSTP requires [channels={channels}, frames>=2], got {} values and {frames} frames",
            input.len()
        )));
    }
    if mask.is_empty() {
        return Err(VokraError::InvalidArgument(
            "wespeaker: weighted TSTP mask must contain at least one frame".into(),
        ));
    }
    if let Some((index, value)) = mask
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(VokraError::InvalidArgument(format!(
            "wespeaker: weighted TSTP mask[{index}] is {value}; expected a finite value in [0, 1]"
        )));
    }

    // PyTorch `F.interpolate(..., mode="nearest")` maps output index `i` to
    // floor(i * input_len / output_len). Keep the interpolated weights once;
    // every feature channel shares the same local-speaker mask.
    let weights: Vec<f32> = (0..frames)
        .map(|frame| mask[(frame * mask.len() / frames).min(mask.len() - 1)])
        .collect();
    let v1 = weights.iter().copied().sum::<f32>() + WEIGHTED_STATS_EPS;
    let v2 = weights.iter().map(|weight| weight * weight).sum::<f32>();
    let denominator = v1 - v2 / v1 + WEIGHTED_STATS_EPS;

    let mut output = vec![0.0; channels * 2];
    for (channel, row) in input.chunks_exact(frames).enumerate() {
        let mean = row
            .iter()
            .zip(weights.iter())
            .map(|(value, weight)| value * weight)
            .sum::<f32>()
            / v1;
        let variance = row
            .iter()
            .zip(weights.iter())
            .map(|(value, weight)| {
                let centered = value - mean;
                centered * centered * weight
            })
            .sum::<f32>()
            / denominator;
        output[channel] = mean;
        output[channels + channel] = variance.sqrt();
    }
    Ok(output)
}

fn detect_layout(file: &GgufFile) -> Result<WeSpeakerArtifactLayout> {
    match file.tensors().len() {
        PREFIXED_TENSOR_COUNT if file.tensor_info("resnet.conv1.weight").is_some() => {
            Ok(WeSpeakerArtifactLayout::PyannotePrefixed)
        }
        BARE_COMBINED_TENSOR_COUNT if file.tensor_info("conv1.weight").is_some() => {
            Ok(WeSpeakerArtifactLayout::OfficialCombinedBare)
        }
        count => Err(VokraError::ModelLoad(format!(
            "wespeaker: unsupported tensor manifest: count={count}; expected exactly {PREFIXED_TENSOR_COUNT} prefixed embedding tensors or {BARE_COMBINED_TENSOR_COUNT} bare combined tensors"
        ))),
    }
}

fn expected_manifest(layout: WeSpeakerArtifactLayout) -> Vec<(String, Vec<usize>)> {
    let stem = layout.stem();
    let include_counter = layout == WeSpeakerArtifactLayout::OfficialCombinedBare;
    let mut expected = Vec::with_capacity(layout.tensor_count());
    push_conv(&mut expected, &format!("{stem}conv1"), 1, 32, 3);
    push_norm(&mut expected, &format!("{stem}bn1"), 32, include_counter);
    let mut input_channels = 32;
    for (stage_index, (&blocks, &output_channels)) in
        STAGE_BLOCKS.iter().zip(&STAGE_CHANNELS).enumerate()
    {
        for block_index in 0..blocks {
            let prefix = format!("{stem}layer{}.{}", stage_index + 1, block_index);
            push_conv(
                &mut expected,
                &format!("{prefix}.conv1"),
                input_channels,
                output_channels,
                3,
            );
            push_norm(
                &mut expected,
                &format!("{prefix}.bn1"),
                output_channels,
                include_counter,
            );
            push_conv(
                &mut expected,
                &format!("{prefix}.conv2"),
                output_channels,
                output_channels,
                3,
            );
            push_norm(
                &mut expected,
                &format!("{prefix}.bn2"),
                output_channels,
                include_counter,
            );
            if stage_index > 0 && block_index == 0 {
                push_conv(
                    &mut expected,
                    &format!("{prefix}.shortcut.0"),
                    input_channels,
                    output_channels,
                    1,
                );
                push_norm(
                    &mut expected,
                    &format!("{prefix}.shortcut.1"),
                    output_channels,
                    include_counter,
                );
            }
            input_channels = output_channels;
        }
    }
    expected.push((format!("{stem}seg_1.weight"), vec![EMBED_DIM, STATS_DIM]));
    expected.push((format!("{stem}seg_1.bias"), vec![EMBED_DIM]));
    if layout == WeSpeakerArtifactLayout::OfficialCombinedBare {
        expected.push(("projection.weight".into(), vec![17_982, EMBED_DIM]));
    }
    expected
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
        vec![output_channels, input_channels, kernel, kernel],
    ));
}

fn push_norm(
    expected: &mut Vec<(String, Vec<usize>)>,
    prefix: &str,
    channels: usize,
    include_counter: bool,
) {
    for suffix in ["weight", "bias", "running_mean", "running_var"] {
        expected.push((format!("{prefix}.{suffix}"), vec![channels]));
    }
    if include_counter {
        expected.push((format!("{prefix}.num_batches_tracked"), Vec::new()));
    }
}

fn verify_manifest(file: &GgufFile, layout: WeSpeakerArtifactLayout) -> Result<()> {
    if file.tensors().len() != layout.tensor_count() {
        return Err(VokraError::ModelLoad(format!(
            "wespeaker: tensor count is {}, expected exactly {} for {}",
            file.tensors().len(),
            layout.tensor_count(),
            layout.contract_name()
        )));
    }
    let expected = expected_manifest(layout);
    debug_assert_eq!(expected.len(), layout.tensor_count());
    for (name, dimensions) in expected {
        check_dims(file, &name, &dimensions)?;
    }
    Ok(())
}

fn check_dims(file: &GgufFile, name: &str, expected: &[usize]) -> Result<()> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("wespeaker: missing tensor `{name}`")))?;
    let expected = expected
        .iter()
        .map(|&dimension| dimension as u64)
        .collect::<Vec<_>>();
    if info.dimensions != expected {
        return Err(VokraError::ModelLoad(format!(
            "wespeaker: tensor `{name}` has dims {:?}, expected {expected:?}",
            info.dimensions
        )));
    }
    Ok(())
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    check_dims(file, name, expected)?;
    file.tensor_f32(name)
        .map_err(|error| VokraError::ModelLoad(format!("wespeaker: reading `{name}`: {error}")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| VokraError::ModelLoad(format!("wespeaker: missing string `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "wespeaker: `{key}` is `{actual}`, expected `{expected}`"
        )));
    }
    Ok(())
}

fn verify_optional_contract(file: &GgufFile, layout: WeSpeakerArtifactLayout) -> Result<()> {
    if !CONTRACT_KEYS.iter().any(|key| file.get(key).is_some()) {
        return Ok(());
    }
    require_string(file, KEY_UPSTREAM_REVISION, UPSTREAM_REVISION)?;
    require_string(file, KEY_SOURCE_REVISION, SOURCE_REVISION)?;
    for (key, expected) in [
        (KEY_SAMPLE_RATE, 16_000),
        (KEY_N_MELS, INPUT_DIM as u32),
        (KEY_FRAME_LENGTH, 400),
        (KEY_FRAME_SHIFT, 160),
        (KEY_EMBED_DIM, EMBED_DIM as u32),
        (KEY_STAGE_COUNT, STAGE_BLOCKS.len() as u32),
    ] {
        require_u32(file, key, expected)?;
    }
    require_f32(file, KEY_BN_EPS, BN_EPS)?;
    require_f32(file, KEY_STATS_EPS, STATS_EPS)?;
    require_string(file, KEY_FRONTEND, "kaldi-fbank-hamming-cmn-v1")?;
    require_string(file, KEY_BLOCKS, "3,4,6,3")?;
    require_string(file, KEY_CHANNELS, "32,64,128,256")?;
    require_string(file, KEY_POOLING, "tstp-bessel-v1")?;
    require_string(file, KEY_LAYOUT, layout.contract_name())
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(|value| value.as_u64())
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| VokraError::ModelLoad(format!("wespeaker: missing u32 `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "wespeaker: `{key}` is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_f32(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    let actual = match file.get(key) {
        Some(GgufMetadataValue::F32(value)) => *value,
        _ => {
            return Err(VokraError::ModelLoad(format!(
                "wespeaker: missing f32 `{key}`"
            )));
        }
    };
    if actual.to_bits() != expected.to_bits() {
        return Err(VokraError::ModelLoad(format!(
            "wespeaker: `{key}` is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_public_manifests_are_exact_and_unique() {
        for layout in [
            WeSpeakerArtifactLayout::PyannotePrefixed,
            WeSpeakerArtifactLayout::OfficialCombinedBare,
        ] {
            let manifest = expected_manifest(layout);
            assert_eq!(manifest.len(), layout.tensor_count());
            let mut names = manifest.iter().map(|(name, _)| name).collect::<Vec<_>>();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), layout.tensor_count());
        }
    }

    #[test]
    fn temporal_pool_uses_bessel_variance_and_epsilon() {
        let pooled = temporal_statistics_pool(&[1.0, 3.0], 1, 2).unwrap();
        assert_eq!(pooled[0], 2.0);
        assert!((pooled[1] - (2.0 + STATS_EPS).sqrt()).abs() < 1.0e-6);
    }

    #[test]
    fn weighted_pool_matches_official_unbiased_formula() {
        let pooled = weighted_temporal_statistics_pool(&[1.0, 3.0], 1, 2, &[1.0, 1.0])
            .expect("two active frames");
        assert_eq!(pooled[0], 2.0);
        assert!((pooled[1] - 2.0f32.sqrt()).abs() < 1.0e-6);
    }

    #[test]
    fn weighted_pool_uses_pytorch_nearest_mask_indices() {
        // Resizing three mask frames to two output frames selects source
        // indices floor([0, 1] * 3 / 2) = [0, 1]. Only the first hidden
        // frame therefore contributes.
        let pooled = weighted_temporal_statistics_pool(&[1.0, 100.0], 1, 2, &[1.0, 0.0, 0.0])
            .expect("nearest resize");
        assert_eq!(pooled, vec![1.0, 0.0]);
    }

    #[test]
    fn weighted_pool_all_zero_mask_is_explicit_zero_statistics() {
        let pooled = weighted_temporal_statistics_pool(&[1.0, 3.0], 1, 2, &[0.0, 0.0])
            .expect("upstream permits an inactive mask");
        assert_eq!(pooled, vec![0.0, 0.0]);
    }

    #[test]
    fn weighted_pool_rejects_empty_or_non_probability_masks() {
        assert!(weighted_temporal_statistics_pool(&[1.0, 3.0], 1, 2, &[]).is_err());
        assert!(weighted_temporal_statistics_pool(&[1.0, 3.0], 1, 2, &[1.0, f32::NAN]).is_err());
        assert!(weighted_temporal_statistics_pool(&[1.0, 3.0], 1, 2, &[1.1, 0.0]).is_err());
    }

    #[test]
    fn frontend_contract_is_pinned() {
        let options = frontend_options();
        assert_eq!(options.sample_rate, 16_000);
        assert_eq!(options.num_mel_bins, 80);
        assert_eq!(options.frame_length, 400);
        assert_eq!(options.frame_shift, 160);
        assert!(options.subtract_mean);
    }

    #[test]
    fn identity_constants_are_pinned() {
        assert_eq!(ARCH, "wespeaker");
        assert_eq!(NAME, "wespeaker-voxceleb-resnet34-lm");
        assert_eq!(UPSTREAM_REVISION.len(), 40);
        assert_eq!(SOURCE_REVISION.len(), 40);
        assert_eq!(EMBED_DIM, 256);
    }
}
