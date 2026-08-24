//! Native runtime binder for Charactr Vocos 24 kHz vocoders.
//!
//! Both released checkpoints use an eight-block ConvNeXt-1D backbone and a
//! magnitude/phase iSTFT head.  The mel release uses plain LayerNorm and
//! center-padded iSTFT; the Encodec release uses bandwidth-conditioned
//! AdaLayerNorm and Vocos' custom same-padded iSTFT.  The Rust runtime accepts
//! already-computed channel-major features, so it does not embed PyTorch,
//! torchaudio, or the Encodec neural encoder.

use std::collections::BTreeSet;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::{
    VocosAttrs, VocosBackendOps, VocosBlockWeights, VocosIstftPadding, VocosNormWeights,
    VocosWeights, vocos_decode, vocos_decode_with_ops,
};

use crate::compute::{Compute, HotOp};

/// Architecture tag emitted by the Vocos converter.
pub const ARCH: &str = "vocos";
/// Required variant metadata key.
pub const KEY_VOCOS_VARIANT: &str = "vokra.vocos.variant";
/// Model category.
pub const CATEGORY: &str = "vocoder";
/// Mel checkpoint variant tag.
pub const VARIANT_TAG_MEL24KHZ: &str = "mel_24khz";
/// Encodec checkpoint variant tag.
pub const VARIANT_TAG_ENCODEC24KHZ: &str = "encodec_24khz";

/// Learned operations required for a complete Vocos Metal forward.
pub const VOCOS_HOT_OPS: &[HotOp] = &[
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::LayerNorm,
    HotOp::Gelu,
];

/// Which official Vocos checkpoint topology is loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocosVariant {
    /// `charactr/vocos-mel-24khz`.
    Mel24khz,
    /// `charactr/vocos-encodec-24khz`.
    Encodec24khz,
}

impl VocosVariant {
    /// Converter model name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mel24khz => "vocos-mel-24khz",
            Self::Encodec24khz => "vocos-encodec-24khz",
        }
    }

    /// Official Hugging Face repository.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Mel24khz => "charactr/vocos-mel-24khz",
            Self::Encodec24khz => "charactr/vocos-encodec-24khz",
        }
    }

    /// GGUF metadata tag.
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Mel24khz => VARIANT_TAG_MEL24KHZ,
            Self::Encodec24khz => VARIANT_TAG_ENCODEC24KHZ,
        }
    }

    /// Parses a GGUF metadata tag without a silent default.
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            VARIANT_TAG_MEL24KHZ => Some(Self::Mel24khz),
            VARIANT_TAG_ENCODEC24KHZ => Some(Self::Encodec24khz),
            _ => None,
        }
    }
}

/// Primary-source-pinned topology for a released Vocos variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocosConfig {
    /// Checkpoint variant.
    pub variant: VocosVariant,
    /// Output sample rate.
    pub sample_rate: u32,
    /// Input feature channels.
    pub n_input: usize,
    /// Backbone hidden width.
    pub dim: usize,
    /// ConvNeXt pointwise hidden width.
    pub intermediate_dim: usize,
    /// ConvNeXt block count.
    pub num_layers: usize,
    /// AdaLayerNorm embedding rows; zero means plain LayerNorm.
    pub num_conditions: usize,
    /// iSTFT FFT size.
    pub n_fft: usize,
    /// iSTFT hop.
    pub hop_length: usize,
    /// iSTFT padding convention.
    pub padding: VocosIstftPadding,
}

impl VocosConfig {
    /// Returns the exact official `config.yaml` axes for `variant`.
    pub const fn for_variant(variant: VocosVariant) -> Self {
        match variant {
            VocosVariant::Mel24khz => Self {
                variant,
                sample_rate: 24_000,
                n_input: 100,
                dim: 512,
                intermediate_dim: 1536,
                num_layers: 8,
                num_conditions: 0,
                n_fft: 1024,
                hop_length: 256,
                padding: VocosIstftPadding::Center,
            },
            VocosVariant::Encodec24khz => Self {
                variant,
                sample_rate: 24_000,
                n_input: 128,
                dim: 384,
                intermediate_dim: 1152,
                num_layers: 8,
                num_conditions: 4,
                n_fft: 1280,
                hop_length: 320,
                padding: VocosIstftPadding::Same,
            },
        }
    }

    fn op_attrs(&self) -> VocosAttrs {
        VocosAttrs {
            input_channels: self.n_input,
            dim: self.dim,
            intermediate_dim: self.intermediate_dim,
            num_layers: self.num_layers,
            num_conditions: self.num_conditions,
            n_fft: self.n_fft,
            hop_length: self.hop_length,
            padding: self.padding,
        }
    }
}

/// A strict real-weight Vocos feature decoder.
#[derive(Debug, Clone)]
pub struct Vocos {
    config: VocosConfig,
    weights: VocosWeights,
    /// Encodec RVQ tables `[16 * 1024, 128]`; absent for the mel model.
    codebook_weights: Option<Vec<f32>>,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl Vocos {
    /// Constructs a Vocos decoder from a complete validated weight bundle.
    pub fn new(variant: VocosVariant, weights: VocosWeights) -> Result<Self> {
        let config = VocosConfig::for_variant(variant);
        weights.validate(&config.op_attrs())?;
        Ok(Self {
            config,
            weights,
            codebook_weights: None,
            weight_license: LicenseClass::Unknown,
            backend: BackendKind::Cpu,
        })
    }

    /// Builds a full zero-weight fixture with official shapes.
    #[must_use]
    pub fn synthesized(variant: VocosVariant) -> Self {
        let config = VocosConfig::for_variant(variant);
        Self::new(variant, synthesized_weights(&config))
            .expect("official Vocos shapes form a valid weight bundle")
    }

    /// Bound topology.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &VocosConfig {
        &self.config
    }

    /// Checkpoint variant.
    #[inline]
    #[must_use]
    pub const fn variant(&self) -> VocosVariant {
        self.config.variant
    }

    /// Output sample rate (24 kHz for both releases).
    #[inline]
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Stamped artifact license class, or `Unknown` when absent.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Selects the backend used by every learned Vocos operation.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Selected inference backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Decodes mel features.  The Encodec variant must use
    /// [`Self::decode_with_bandwidth`] because its AdaLayerNorm condition is
    /// a required part of the official contract.
    pub fn decode(&self, features: &[f32], n_frames: usize) -> Result<Vec<f32>> {
        if self.variant() == VocosVariant::Encodec24khz {
            return Err(VokraError::InvalidArgument(
                "Vocos::decode(encodec_24khz): bandwidth_id is required; use decode_with_bandwidth(features, n_frames, 0..4)"
                    .to_owned(),
            ));
        }
        self.decode_inner(features, n_frames, None)
    }

    /// Decodes Encodec features with the official bandwidth condition id:
    /// `0,1,2,3` map to `1.5,3.0,6.0,12.0` kbps respectively.
    pub fn decode_with_bandwidth(
        &self,
        features: &[f32],
        n_frames: usize,
        bandwidth_id: usize,
    ) -> Result<Vec<f32>> {
        if self.variant() != VocosVariant::Encodec24khz {
            return Err(VokraError::InvalidArgument(
                "Vocos::decode_with_bandwidth is only valid for encodec_24khz".to_owned(),
            ));
        }
        self.decode_inner(features, n_frames, Some(bandwidth_id))
    }

    fn decode_inner(
        &self,
        features: &[f32],
        n_frames: usize,
        bandwidth_id: Option<usize>,
    ) -> Result<Vec<f32>> {
        let attrs = self.config.op_attrs();
        if self.backend == BackendKind::Cpu {
            return vocos_decode(features, n_frames, bandwidth_id, &self.weights, &attrs);
        }
        let compute = Compute::for_backend(self.backend, VOCOS_HOT_OPS)?;
        let ops = ComputeVocosOps { compute: &compute };
        vocos_decode_with_ops(
            features,
            n_frames,
            bandwidth_id,
            &self.weights,
            &attrs,
            &ops,
        )
    }

    /// Sums Encodec RVQ embeddings into channel-major `[128, frames]`
    /// features, matching upstream `Vocos.codes_to_features`.
    pub fn codes_to_features(
        &self,
        codes: &[u32],
        n_codebooks: usize,
        n_frames: usize,
    ) -> Result<Vec<f32>> {
        const BINS: usize = 1024;
        const DIM: usize = 128;
        const MAX_CODEBOOKS: usize = 16;
        let tables = self.codebook_weights.as_ref().ok_or_else(|| {
            VokraError::InvalidArgument(
                "Vocos::codes_to_features is only available on encodec_24khz".to_owned(),
            )
        })?;
        if n_codebooks == 0 || n_codebooks > MAX_CODEBOOKS || n_frames == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "Vocos::codes_to_features: expected 1..={MAX_CODEBOOKS} codebooks and positive frames, got {n_codebooks}x{n_frames}"
            )));
        }
        if codes.len() != n_codebooks * n_frames {
            return Err(VokraError::InvalidArgument(format!(
                "Vocos::codes_to_features: codes has length {}, expected {}",
                codes.len(),
                n_codebooks * n_frames
            )));
        }
        let mut features = vec![0.0f32; DIM * n_frames];
        for codebook in 0..n_codebooks {
            for frame in 0..n_frames {
                let code = codes[codebook * n_frames + frame] as usize;
                if code >= BINS {
                    return Err(VokraError::InvalidArgument(format!(
                        "Vocos::codes_to_features: code {code} at codebook {codebook}, frame {frame} is outside 0..{BINS}"
                    )));
                }
                let row = (codebook * BINS + code) * DIM;
                for channel in 0..DIM {
                    features[channel * n_frames + frame] += tables[row + channel];
                }
            }
        }
        Ok(features)
    }

    /// Loads every tensor from a converted official Vocos GGUF, validates
    /// exact shapes and rejects missing or extra manifest entries.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let arch = file
            .get(chunks::KEY_MODEL_ARCH)
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "Vocos::from_gguf: missing or non-string `{}`",
                    chunks::KEY_MODEL_ARCH
                ))
            })?;
        if arch != ARCH {
            return Err(VokraError::ModelLoad(format!(
                "Vocos::from_gguf: arch {arch:?}, expected {ARCH:?}"
            )));
        }
        let tag = file
            .get(KEY_VOCOS_VARIANT)
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "Vocos::from_gguf: missing `{KEY_VOCOS_VARIANT}`; expected {VARIANT_TAG_MEL24KHZ:?} or {VARIANT_TAG_ENCODEC24KHZ:?}"
                ))
            })?;
        let variant = VocosVariant::from_tag(tag).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "Vocos::from_gguf: unknown variant {tag:?}; expected {VARIANT_TAG_MEL24KHZ:?} or {VARIANT_TAG_ENCODEC24KHZ:?}"
            ))
        })?;
        let config = VocosConfig::for_variant(variant);
        let (weights, codebook_weights) = load_weights(file, &config)?;
        let mut model = Self::new(variant, weights).map_err(|error| {
            VokraError::ModelLoad(format!(
                "Vocos::from_gguf: weight validation failed: {error}"
            ))
        })?;
        model.codebook_weights = codebook_weights;
        model.weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|value| value.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        Ok(model)
    }
}

struct ComputeVocosOps<'a> {
    compute: &'a Compute,
}

impl VocosBackendOps for ComputeVocosOps<'_> {
    fn conv1d_same(
        &self,
        input: &[f32],
        input_channels: usize,
        frames: usize,
        output_channels: usize,
        weight: &[f32],
        bias: &[f32],
        kernel: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; output_channels * frames];
        self.compute.conv1d_f32(
            input,
            input_channels,
            frames,
            weight,
            output_channels,
            kernel,
            Some(bias),
            1,
            kernel / 2,
            &mut output,
        )?;
        Ok(output)
    }

    fn depthwise_conv1d_same(
        &self,
        input: &[f32],
        channels: usize,
        frames: usize,
        weight: &[f32],
        bias: &[f32],
        kernel: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; channels * frames];
        self.compute.grouped_conv1d_f32(
            input,
            channels,
            frames,
            weight,
            channels,
            kernel,
            Some(bias),
            1,
            kernel / 2,
            channels,
            &mut output,
        )?;
        Ok(output)
    }

    fn norm_channel_major(
        &self,
        values: &mut [f32],
        frames: usize,
        dim: usize,
        scale: &[f32],
        shift: &[f32],
    ) -> Result<()> {
        let mut frame_major = vec![0.0f32; values.len()];
        for channel in 0..dim {
            for frame in 0..frames {
                frame_major[frame * dim + channel] = values[channel * frames + frame];
            }
        }
        let mut normalized = vec![0.0f32; values.len()];
        self.compute.layer_norm_f32(
            &frame_major,
            &mut normalized,
            frames,
            dim,
            scale,
            shift,
            1e-6,
        )?;
        for channel in 0..dim {
            for frame in 0..frames {
                values[channel * frames + frame] = normalized[frame * dim + channel];
            }
        }
        Ok(())
    }

    fn pointwise(
        &self,
        input: &[f32],
        input_dim: usize,
        frames: usize,
        output_dim: usize,
        weight: &[f32],
        bias: &[f32],
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; output_dim * frames];
        self.compute.conv1d_f32(
            input,
            input_dim,
            frames,
            weight,
            output_dim,
            1,
            Some(bias),
            1,
            0,
            &mut output,
        )?;
        Ok(output)
    }

    fn gelu_in_place(&self, values: &mut [f32]) -> Result<()> {
        let mut output = vec![0.0f32; values.len()];
        self.compute.gelu_f32(values, &mut output)?;
        values.copy_from_slice(&output);
        Ok(())
    }
}

fn load_weights(file: &GgufFile, config: &VocosConfig) -> Result<(VocosWeights, Option<Vec<f32>>)> {
    let mut expected = BTreeSet::new();
    let dim = config.dim;
    let intermediate = config.intermediate_dim;
    let embed_weight = load_tensor(
        file,
        "backbone.embed.weight",
        &[dim, config.n_input, 7],
        &mut expected,
    )?;
    let embed_bias = load_tensor(file, "backbone.embed.bias", &[dim], &mut expected)?;
    let norm = load_norm(file, "backbone.norm", config, &mut expected)?;
    let mut blocks = Vec::with_capacity(config.num_layers);
    for index in 0..config.num_layers {
        let prefix = format!("backbone.convnext.{index}");
        blocks.push(VocosBlockWeights {
            depthwise_weight: load_tensor(
                file,
                &format!("{prefix}.dwconv.weight"),
                &[dim, 1, 7],
                &mut expected,
            )?,
            depthwise_bias: load_tensor(
                file,
                &format!("{prefix}.dwconv.bias"),
                &[dim],
                &mut expected,
            )?,
            norm: load_norm(file, &format!("{prefix}.norm"), config, &mut expected)?,
            pointwise1_weight: load_tensor(
                file,
                &format!("{prefix}.pwconv1.weight"),
                &[intermediate, dim],
                &mut expected,
            )?,
            pointwise1_bias: load_tensor(
                file,
                &format!("{prefix}.pwconv1.bias"),
                &[intermediate],
                &mut expected,
            )?,
            pointwise2_weight: load_tensor(
                file,
                &format!("{prefix}.pwconv2.weight"),
                &[dim, intermediate],
                &mut expected,
            )?,
            pointwise2_bias: load_tensor(
                file,
                &format!("{prefix}.pwconv2.bias"),
                &[dim],
                &mut expected,
            )?,
            gamma: load_tensor(file, &format!("{prefix}.gamma"), &[dim], &mut expected)?,
        });
    }
    let final_norm_weight = load_tensor(
        file,
        "backbone.final_layer_norm.weight",
        &[dim],
        &mut expected,
    )?;
    let final_norm_bias = load_tensor(
        file,
        "backbone.final_layer_norm.bias",
        &[dim],
        &mut expected,
    )?;
    let head_weight = load_tensor(
        file,
        "head.out.weight",
        &[config.n_fft + 2, dim],
        &mut expected,
    )?;
    let head_bias = load_tensor(file, "head.out.bias", &[config.n_fft + 2], &mut expected)?;
    let _istft_window = load_tensor(file, "head.istft.window", &[config.n_fft], &mut expected)?;

    let codebook_weights = match config.variant {
        VocosVariant::Mel24khz => {
            let _mel_filterbank = load_tensor(
                file,
                "feature_extractor.mel_spec.mel_scale.fb",
                &[513, 100],
                &mut expected,
            )?;
            let _mel_window = load_tensor(
                file,
                "feature_extractor.mel_spec.spectrogram.window",
                &[1024],
                &mut expected,
            )?;
            None
        }
        VocosVariant::Encodec24khz => Some(load_tensor(
            file,
            "feature_extractor.codebook_weights",
            &[16_384, 128],
            &mut expected,
        )?),
    };

    let actual: BTreeSet<String> = file
        .tensors()
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect();
    if actual != expected {
        let missing: Vec<&String> = expected.difference(&actual).take(4).collect();
        let extra: Vec<&String> = actual.difference(&expected).take(4).collect();
        return Err(VokraError::ModelLoad(format!(
            "Vocos: tensor manifest mismatch (expected {}, found {}); missing={missing:?}, extra={extra:?}",
            expected.len(),
            actual.len()
        )));
    }

    Ok((
        VocosWeights {
            embed_weight,
            embed_bias,
            norm,
            blocks,
            final_norm_weight,
            final_norm_bias,
            head_weight,
            head_bias,
        },
        codebook_weights,
    ))
}

fn load_norm(
    file: &GgufFile,
    prefix: &str,
    config: &VocosConfig,
    expected: &mut BTreeSet<String>,
) -> Result<VocosNormWeights> {
    if config.num_conditions == 0 {
        Ok(VocosNormWeights {
            scale: load_tensor(file, &format!("{prefix}.weight"), &[config.dim], expected)?,
            shift: load_tensor(file, &format!("{prefix}.bias"), &[config.dim], expected)?,
        })
    } else {
        Ok(VocosNormWeights {
            scale: load_tensor(
                file,
                &format!("{prefix}.scale.weight"),
                &[config.num_conditions, config.dim],
                expected,
            )?,
            shift: load_tensor(
                file,
                &format!("{prefix}.shift.weight"),
                &[config.num_conditions, config.dim],
                expected,
            )?,
        })
    }
}

fn load_tensor(
    file: &GgufFile,
    name: &str,
    expected_shape: &[usize],
    expected_names: &mut BTreeSet<String>,
) -> Result<Vec<f32>> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("Vocos: required tensor `{name}` is missing"))
    })?;
    let actual_shape: Vec<usize> = info
        .dimensions
        .iter()
        .map(|&dimension| dimension as usize)
        .collect();
    if actual_shape != expected_shape {
        return Err(VokraError::ModelLoad(format!(
            "Vocos: tensor `{name}` shape {actual_shape:?}, expected {expected_shape:?}"
        )));
    }
    expected_names.insert(name.to_owned());
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("Vocos: tensor `{name}` decode failed: {error}"))
    })
}

fn synthesized_weights(config: &VocosConfig) -> VocosWeights {
    let norm = || VocosNormWeights {
        scale: vec![1.0; config.num_conditions.max(1) * config.dim],
        shift: vec![0.0; config.num_conditions.max(1) * config.dim],
    };
    VocosWeights {
        embed_weight: vec![0.0; config.dim * config.n_input * 7],
        embed_bias: vec![0.0; config.dim],
        norm: norm(),
        blocks: (0..config.num_layers)
            .map(|_| VocosBlockWeights {
                depthwise_weight: vec![0.0; config.dim * 7],
                depthwise_bias: vec![0.0; config.dim],
                norm: norm(),
                pointwise1_weight: vec![0.0; config.intermediate_dim * config.dim],
                pointwise1_bias: vec![0.0; config.intermediate_dim],
                pointwise2_weight: vec![0.0; config.dim * config.intermediate_dim],
                pointwise2_bias: vec![0.0; config.dim],
                gamma: vec![0.125; config.dim],
            })
            .collect(),
        final_norm_weight: vec![1.0; config.dim],
        final_norm_bias: vec![0.0; config.dim],
        head_weight: vec![0.0; (config.n_fft + 2) * config.dim],
        head_bias: vec![0.0; config.n_fft + 2],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    #[test]
    fn official_config_axes_are_pinned() {
        let mel = VocosConfig::for_variant(VocosVariant::Mel24khz);
        assert_eq!(
            (mel.n_input, mel.dim, mel.intermediate_dim),
            (100, 512, 1536)
        );
        assert_eq!(
            (mel.n_fft, mel.hop_length, mel.num_conditions),
            (1024, 256, 0)
        );
        assert_eq!(mel.padding, VocosIstftPadding::Center);

        let encodec = VocosConfig::for_variant(VocosVariant::Encodec24khz);
        assert_eq!(
            (encodec.n_input, encodec.dim, encodec.intermediate_dim),
            (128, 384, 1152)
        );
        assert_eq!(
            (encodec.n_fft, encodec.hop_length, encodec.num_conditions),
            (1280, 320, 4)
        );
        assert_eq!(encodec.padding, VocosIstftPadding::Same);
    }

    #[test]
    fn variant_tags_round_trip_without_default() {
        for variant in [VocosVariant::Mel24khz, VocosVariant::Encodec24khz] {
            assert_eq!(VocosVariant::from_tag(variant.tag()), Some(variant));
        }
        assert_eq!(VocosVariant::from_tag("mel_48khz"), None);
    }

    #[test]
    fn encodec_decode_requires_bandwidth_condition() {
        let model = Vocos::synthesized(VocosVariant::Encodec24khz);
        let err = model.decode(&vec![0.0; 128 * 2], 2).unwrap_err();
        assert!(err.to_string().contains("bandwidth_id"));
    }

    #[test]
    fn loader_rejects_incomplete_manifest() {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        builder.add_string(KEY_VOCOS_VARIANT, VARIANT_TAG_MEL24KHZ);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let err = Vocos::from_gguf(&file).unwrap_err();
        assert!(err.to_string().contains("backbone.embed.weight"));
    }

    #[test]
    fn loader_rejects_unknown_variant() {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        builder.add_string(KEY_VOCOS_VARIANT, "future");
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let err = Vocos::from_gguf(&file).unwrap_err();
        assert!(err.to_string().contains("unknown variant"));
    }
}
