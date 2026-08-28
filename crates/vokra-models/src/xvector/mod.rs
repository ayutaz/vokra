//! Native SpeechBrain X-vector speaker embedding.
//!
//! This module implements the released `speechbrain/spkrec-xvect-voxceleb`
//! topology: 24-bin SpeechBrain fbank, five reflect-padded TDNN layers with
//! LeakyReLU and eval-mode BatchNorm, statistics pooling, and a 512-dimensional
//! projection.  The runtime accepts both public Vokra layouts deliberately:
//! the historical 32-tensor embedding-only file and the 46-tensor combined
//! file whose embedding tensors carry an `embedding_model.` prefix.
//!
//! SpeechBrain's statistics pool adds tiny random noise to the mean even in
//! inference.  Vokra omits that non-deterministic dither; the independent
//! SpeechBrain parity fixture gates the resulting bounded difference while
//! making repeated native inference deterministic.

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{BackendKind, LicenseClass, Result, SpeakerEngine, VokraError};
use vokra_ops::{SpeechbrainFbankAttrs, speechbrain_fbank};

use crate::compute::{Compute, HotOp};

/// `vokra.model.arch` accepted by this runtime.
pub const ARCH: &str = "xvector";
/// Canonical model identity stamped by both public artifacts.
pub const NAME: &str = "spkrec-xvect-voxceleb";
/// Model task category.
pub const CATEGORY: &str = "speaker";
/// Pinned upstream Hugging Face repository.
pub const UPSTREAM_HF: &str = "speechbrain/spkrec-xvect-voxceleb";
/// Audited upstream repository revision for future conversions.
pub const UPSTREAM_REVISION: &str = "56895a2df401be4150a159f3a1c653f00051d477";
/// Output speaker-embedding width.
pub const EMBED_DIM: usize = 512;

const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_SAMPLE_RATE: &str = "vokra.xvector.sample_rate";
const KEY_N_MELS: &str = "vokra.xvector.n_mels";
const KEY_N_FFT: &str = "vokra.xvector.n_fft";
const KEY_WIN_LENGTH: &str = "vokra.xvector.win_length";
const KEY_HOP_LENGTH: &str = "vokra.xvector.hop_length";
const KEY_EMBED_DIM: &str = "vokra.xvector.embed_dim";
const KEY_TDNN_BLOCKS: &str = "vokra.xvector.tdnn_blocks";
const KEY_BN_EPS: &str = "vokra.xvector.bn_eps";
const KEY_STATS_STD_EPS: &str = "vokra.xvector.stats_std_eps";
const KEY_FRONTEND: &str = "vokra.xvector.frontend";
const KEY_PADDING: &str = "vokra.xvector.padding";
const KEY_ARTIFACT_LAYOUT: &str = "vokra.xvector.artifact_layout";
const CONTRACT_KEYS: [&str; 12] = [
    KEY_SAMPLE_RATE,
    KEY_N_MELS,
    KEY_N_FFT,
    KEY_WIN_LENGTH,
    KEY_HOP_LENGTH,
    KEY_EMBED_DIM,
    KEY_TDNN_BLOCKS,
    KEY_BN_EPS,
    KEY_STATS_STD_EPS,
    KEY_FRONTEND,
    KEY_PADDING,
    KEY_ARTIFACT_LAYOUT,
];
const INPUT_DIM: usize = 24;
const STATS_CHANNELS: usize = 1_500;
const STATS_DIM: usize = STATS_CHANNELS * 2;
const BN_EPS: f32 = 1.0e-5;
const LEAKY_RELU_SLOPE: f32 = 0.01;
const STATS_STD_EPS: f32 = 1.0e-5;
const EMBEDDING_TENSORS: usize = 32;
const COMBINED_TENSORS: usize = 46;
const XVECTOR_HOT_OPS: &[HotOp] = &[HotOp::Conv1d];

#[derive(Debug, Clone, Copy)]
struct TdnnSpec {
    block: usize,
    norm_block: usize,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    dilation: usize,
}

const TDNN_SPECS: [TdnnSpec; 5] = [
    TdnnSpec {
        block: 0,
        norm_block: 2,
        input_channels: 24,
        output_channels: 512,
        kernel: 5,
        dilation: 1,
    },
    TdnnSpec {
        block: 3,
        norm_block: 5,
        input_channels: 512,
        output_channels: 512,
        kernel: 3,
        dilation: 2,
    },
    TdnnSpec {
        block: 6,
        norm_block: 8,
        input_channels: 512,
        output_channels: 512,
        kernel: 3,
        dilation: 3,
    },
    TdnnSpec {
        block: 9,
        norm_block: 11,
        input_channels: 512,
        output_channels: 512,
        kernel: 1,
        dilation: 1,
    },
    TdnnSpec {
        block: 12,
        norm_block: 14,
        input_channels: 512,
        output_channels: 1_500,
        kernel: 1,
        dilation: 1,
    },
];

/// Which exact public tensor layout was bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XVectorArtifactLayout {
    /// 32 embedding tensors named `blocks.*`.
    EmbeddingOnlyBare,
    /// 32 `embedding_model.blocks.*` tensors plus 14 classifier/norm tensors.
    CombinedPrefixed,
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

    fn apply(&self, values: &mut [f32], frames: usize) {
        debug_assert_eq!(values.len(), self.scale.len() * frames);
        for channel in 0..self.scale.len() {
            let row = &mut values[channel * frames..(channel + 1) * frames];
            let scale = self.scale[channel];
            let shift = self.shift[channel];
            for value in row {
                *value = *value * scale + shift;
            }
        }
    }
}

#[derive(Debug)]
struct TdnnLayer {
    weight: Vec<f32>,
    bias: Vec<f32>,
    norm: FoldedBatchNorm,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    dilation: usize,
}

impl TdnnLayer {
    fn bind(file: &GgufFile, stem: &str, spec: TdnnSpec) -> Result<Self> {
        let conv = format!("{stem}blocks.{}.conv", spec.block);
        let norm = format!("{stem}blocks.{}.norm", spec.norm_block);
        Ok(Self {
            weight: tensor(
                file,
                &format!("{conv}.weight"),
                &[spec.output_channels, spec.input_channels, spec.kernel],
            )?,
            bias: tensor(file, &format!("{conv}.bias"), &[spec.output_channels])?,
            norm: FoldedBatchNorm::bind(file, &norm, spec.output_channels)?,
            input_channels: spec.input_channels,
            output_channels: spec.output_channels,
            kernel: spec.kernel,
            dilation: spec.dilation,
        })
    }

    fn forward(&self, input: &[f32], frames: usize, compute: &Compute) -> Result<Vec<f32>> {
        if input.len() != self.input_channels * frames {
            return Err(VokraError::InvalidArgument(format!(
                "xvector: TDNN input has {} values, expected {} x {frames}",
                input.len(),
                self.input_channels
            )));
        }
        let effective_kernel = (self.kernel - 1) * self.dilation + 1;
        let padding = effective_kernel / 2;
        let padded = reflect_pad(input, self.input_channels, frames, padding)?;
        let padded_frames = frames + 2 * padding;
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
            padded_frames,
            &expanded_weight,
            self.output_channels,
            effective_kernel,
            Some(&self.bias),
            1,
            0,
            &mut output,
        )?;
        for value in &mut output {
            if *value < 0.0 {
                *value *= LEAKY_RELU_SLOPE;
            }
        }
        self.norm.apply(&mut output, frames);
        Ok(output)
    }
}

#[derive(Debug)]
struct XVectorWeights {
    layers: Vec<TdnnLayer>,
    projection_weight: Vec<f32>,
    projection_bias: Vec<f32>,
    layout: XVectorArtifactLayout,
    tensor_count: usize,
}

impl XVectorWeights {
    fn bind(file: &GgufFile) -> Result<Self> {
        let (layout, stem) = detect_layout(file)?;
        verify_manifest(file, layout, stem)?;
        let layers = TDNN_SPECS
            .iter()
            .copied()
            .map(|spec| TdnnLayer::bind(file, stem, spec))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            projection_weight: tensor(
                file,
                &format!("{stem}blocks.16.w.weight"),
                &[EMBED_DIM, STATS_DIM],
            )?,
            projection_bias: tensor(file, &format!("{stem}blocks.16.w.bias"), &[EMBED_DIM])?,
            layout,
            tensor_count: file.tensors().len(),
        })
    }

    fn embed_features(
        &self,
        features: &[f32],
        frames: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if features.len() != frames * INPUT_DIM {
            return Err(VokraError::InvalidArgument(format!(
                "xvector: feature buffer has {} values, expected {frames} x {INPUT_DIM}",
                features.len()
            )));
        }
        if frames <= 3 {
            return Err(VokraError::InvalidArgument(format!(
                "xvector: {frames} feature frames are too short for dilation-3 reflect padding"
            )));
        }
        let mut channel_major = vec![0.0; INPUT_DIM * frames];
        for frame in 0..frames {
            for channel in 0..INPUT_DIM {
                channel_major[channel * frames + frame] = features[frame * INPUT_DIM + channel];
            }
        }
        let mut hidden = channel_major;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, frames, compute)?;
        }
        let pooled = statistics_pool(&hidden, STATS_CHANNELS, frames)?;
        let mut embedding = vec![0.0; EMBED_DIM];
        compute.conv1d_f32(
            &pooled,
            STATS_DIM,
            1,
            &self.projection_weight,
            EMBED_DIM,
            1,
            Some(&self.projection_bias),
            1,
            0,
            &mut embedding,
        )?;
        Ok(embedding)
    }
}

/// Complete native X-vector inference handle.
#[derive(Debug)]
pub struct XVector {
    weights: XVectorWeights,
    frontend: SpeechbrainFbankAttrs,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl XVector {
    /// Strictly binds one of the two exact public X-vector manifests.
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
        let weights = XVectorWeights::bind(file)?;
        verify_optional_contract(file, weights.layout)?;
        Ok(Self {
            weights,
            frontend: SpeechbrainFbankAttrs::xvector_voxceleb(),
            weight_license,
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and strictly binds a GGUF file.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let file = GgufFile::open(path)?;
        Self::from_gguf(&file)
    }

    /// Selects one backend for every learned TDNN/projection convolution.
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

    /// Returns the bound public tensor-layout variant.
    #[must_use]
    pub const fn artifact_layout(&self) -> XVectorArtifactLayout {
        self.weights.layout
    }

    /// Returns the exact tensor count of the bound artifact.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.weights.tensor_count
    }

    /// Returns the normalized provenance license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Runs the PCM frontend plus complete X-vector network.
    pub fn embed_pcm(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        if sample_rate != self.frontend.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "xvector: expected {} Hz mono PCM, got {sample_rate} Hz; resample offline first",
                self.frontend.sample_rate
            )));
        }
        let (features, frames) = speechbrain_fbank(pcm, &self.frontend)?;
        self.embed_features(&features, frames)
    }

    /// Runs the learned network on row-major `[frames, 24]` frontend features.
    pub fn embed_features(&self, features: &[f32], frames: usize) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, XVECTOR_HOT_OPS)?;
        self.weights.embed_features(features, frames, &compute)
    }

    /// Computes only the official SpeechBrain frontend for parity diagnostics.
    pub fn frontend_features(&self, pcm: &[f32], sample_rate: u32) -> Result<(Vec<f32>, usize)> {
        if sample_rate != self.frontend.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "xvector: expected {} Hz mono PCM, got {sample_rate} Hz",
                self.frontend.sample_rate
            )));
        }
        speechbrain_fbank(pcm, &self.frontend)
    }
}

impl SpeakerEngine for XVector {
    fn embed(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        self.embed_pcm(pcm, sample_rate)
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }
}

fn statistics_pool(input: &[f32], channels: usize, frames: usize) -> Result<Vec<f32>> {
    if input.len() != channels * frames || frames < 2 {
        return Err(VokraError::InvalidArgument(format!(
            "xvector: statistics pool requires [channels={channels}, frames>=2], got {} values and {frames} frames",
            input.len()
        )));
    }
    let mut output = vec![0.0; channels * 2];
    for channel in 0..channels {
        let row = &input[channel * frames..(channel + 1) * frames];
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
        output[channels + channel] = variance.sqrt() + STATS_STD_EPS;
    }
    Ok(output)
}

fn reflect_pad(input: &[f32], channels: usize, frames: usize, padding: usize) -> Result<Vec<f32>> {
    if padding == 0 {
        return Ok(input.to_vec());
    }
    if frames <= padding {
        return Err(VokraError::InvalidArgument(format!(
            "xvector: reflect padding {padding} requires more than {padding} frames, got {frames}"
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

fn detect_layout(file: &GgufFile) -> Result<(XVectorArtifactLayout, &'static str)> {
    let bare = file.tensor_info("blocks.0.conv.weight").is_some();
    let prefixed = file
        .tensor_info("embedding_model.blocks.0.conv.weight")
        .is_some();
    match (bare, prefixed, file.tensors().len()) {
        (true, false, EMBEDDING_TENSORS) => Ok((XVectorArtifactLayout::EmbeddingOnlyBare, "")),
        (false, true, COMBINED_TENSORS) => {
            Ok((XVectorArtifactLayout::CombinedPrefixed, "embedding_model."))
        }
        _ => Err(VokraError::ModelLoad(format!(
            "xvector: unsupported public tensor layout: count={}, bare_stem={bare}, prefixed_stem={prefixed}; expected exactly 32 bare embedding tensors or 46 combined tensors",
            file.tensors().len()
        ))),
    }
}

fn verify_manifest(file: &GgufFile, layout: XVectorArtifactLayout, stem: &str) -> Result<()> {
    for spec in TDNN_SPECS {
        check_dims(
            file,
            &format!("{stem}blocks.{}.conv.weight", spec.block),
            &[spec.output_channels, spec.input_channels, spec.kernel],
        )?;
        check_dims(
            file,
            &format!("{stem}blocks.{}.conv.bias", spec.block),
            &[spec.output_channels],
        )?;
        for suffix in ["weight", "bias", "running_mean", "running_var"] {
            check_dims(
                file,
                &format!("{stem}blocks.{}.norm.{suffix}", spec.norm_block),
                &[spec.output_channels],
            )?;
        }
    }
    check_dims(
        file,
        &format!("{stem}blocks.16.w.weight"),
        &[EMBED_DIM, STATS_DIM],
    )?;
    check_dims(file, &format!("{stem}blocks.16.w.bias"), &[EMBED_DIM])?;
    if layout == XVectorArtifactLayout::CombinedPrefixed {
        for prefix in ["classifier.norm.norm", "classifier.DNN.block_0.norm.norm"] {
            for suffix in ["weight", "bias", "running_mean", "running_var"] {
                check_dims(file, &format!("{prefix}.{suffix}"), &[512])?;
            }
        }
        check_dims(file, "classifier.DNN.block_0.linear.w.weight", &[512, 512])?;
        check_dims(file, "classifier.DNN.block_0.linear.w.bias", &[512])?;
        check_dims(file, "classifier.out.w.weight", &[7_205, 512])?;
        check_dims(file, "classifier.out.w.bias", &[7_205])?;
        check_dims(file, "mean_var_norm_emb.glob_mean", &[512])?;
        check_dims(file, "mean_var_norm_emb.glob_std", &[1])?;
    }
    Ok(())
}

fn check_dims(file: &GgufFile, name: &str, expected: &[usize]) -> Result<()> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("xvector: missing tensor `{name}`")))?;
    let expected = expected
        .iter()
        .map(|&dimension| dimension as u64)
        .collect::<Vec<_>>();
    if info.dimensions != expected {
        return Err(VokraError::ModelLoad(format!(
            "xvector: tensor `{name}` has dims {:?}, expected {expected:?}",
            info.dimensions
        )));
    }
    Ok(())
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    check_dims(file, name, expected)?;
    file.tensor_f32(name)
        .map_err(|error| VokraError::ModelLoad(format!("xvector: reading `{name}`: {error}")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| VokraError::ModelLoad(format!("xvector: missing string `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "xvector: `{key}` is `{actual}`, expected `{expected}`"
        )));
    }
    Ok(())
}

fn verify_optional_contract(file: &GgufFile, layout: XVectorArtifactLayout) -> Result<()> {
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
        (KEY_TDNN_BLOCKS, TDNN_SPECS.len() as u32),
    ] {
        require_u32(file, key, expected)?;
    }
    require_f32(file, KEY_BN_EPS, BN_EPS)?;
    require_f32(file, KEY_STATS_STD_EPS, STATS_STD_EPS)?;
    require_string(file, KEY_FRONTEND, "speechbrain-fbank-v1")?;
    require_string(file, KEY_PADDING, "reflect-same")?;
    require_string(
        file,
        KEY_ARTIFACT_LAYOUT,
        match layout {
            XVectorArtifactLayout::EmbeddingOnlyBare => "embedding-only-bare-v1",
            XVectorArtifactLayout::CombinedPrefixed => "combined-prefixed-v1",
        },
    )
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(|value| value.as_u64())
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| VokraError::ModelLoad(format!("xvector: missing u32 `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "xvector: `{key}` is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_f32(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    let actual = match file.get(key) {
        Some(GgufMetadataValue::F32(value)) => *value,
        _ => {
            return Err(VokraError::ModelLoad(format!(
                "xvector: missing f32 `{key}`"
            )));
        }
    };
    if actual.to_bits() != expected.to_bits() {
        return Err(VokraError::ModelLoad(format!(
            "xvector: `{key}` is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_padding_matches_torch_examples() {
        let padded = reflect_pad(&[1.0, 2.0, 3.0, 4.0], 1, 4, 2).unwrap();
        assert_eq!(padded, [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn dilation_expands_only_kernel_taps() {
        let expanded = expand_dilated_kernel(&[1.0, 2.0, 3.0], 1, 1, 3, 3);
        assert_eq!(expanded, [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn statistics_pool_uses_bessel_std_and_speechbrain_epsilon() {
        let pooled = statistics_pool(&[1.0, 3.0], 1, 2).unwrap();
        assert_eq!(pooled[0], 2.0);
        assert!((pooled[1] - (2.0f32.sqrt() + STATS_STD_EPS)).abs() < 1.0e-6);
    }

    #[test]
    fn public_identity_constants_are_pinned() {
        assert_eq!(ARCH, "xvector");
        assert_eq!(NAME, "spkrec-xvect-voxceleb");
        assert_eq!(UPSTREAM_REVISION.len(), 40);
        assert_eq!(EMBED_DIM, 512);
    }

    #[test]
    fn partial_new_metadata_contract_is_rejected() {
        let mut builder = vokra_core::gguf::GgufBuilder::new();
        builder.add_u32(KEY_SAMPLE_RATE, 16_000);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let error =
            verify_optional_contract(&file, XVectorArtifactLayout::EmbeddingOnlyBare).unwrap_err();
        assert!(error.to_string().contains(KEY_UPSTREAM_REVISION));
    }
}
