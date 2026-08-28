//! Strict native SpeechBrain SepFormer inference.
//!
//! The seven public checkpoints share a 256-channel encoder, two dual-path
//! blocks and a learned decoder.  Each intra/inter branch is an eight-layer
//! pre-norm Transformer.  CPU and Metal execute the same backend-dispatched
//! Conv1D, GEMM, softmax and LayerNorm operations; layout changes, overlap-add
//! and pointwise activations remain host control/DSP glue.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};

/// GGUF architecture tag shared by all supported SepFormer variants.
pub const ARCH: &str = "sepformer";
/// Metadata key identifying separation versus enhancement models.
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// Metadata key holding the concrete SepFormer variant tag.
pub const KEY_SEPFORMER_VARIANT: &str = "vokra.sepformer.variant";
/// Metadata key holding the number of output waveform streams.
pub const KEY_SEPFORMER_N_OUT: &str = "vokra.sepformer.n_out";
/// Metadata value for multi-speaker source-separation checkpoints.
pub const CATEGORY_SEPARATION: &str = "separation";
/// Metadata value for single-stream speech-enhancement checkpoints.
pub const CATEGORY_ENHANCEMENT: &str = "enhancement";
/// Variant tag for the WSJ0 2-speaker checkpoint.
pub const VARIANT_TAG_WSJ02MIX: &str = "wsj02mix";
/// Variant tag for the LibriMix 2-speaker checkpoint.
pub const VARIANT_TAG_LIBRI2MIX: &str = "libri2mix";
/// Variant tag for the LibriMix 3-speaker checkpoint.
pub const VARIANT_TAG_LIBRI3MIX: &str = "libri3mix";
/// Variant tag for the WHAM! 16 kHz enhancement checkpoint.
pub const VARIANT_TAG_WHAM16K_ENHANCEMENT: &str = "wham16k-enhancement";
/// Variant tag for the WHAMR! 16 kHz separation checkpoint.
pub const VARIANT_TAG_WHAMR16K: &str = "whamr16k";
/// Variant tag for the WHAMR! 8 kHz separation checkpoint.
pub const VARIANT_TAG_WHAMR8K: &str = "whamr8k";
/// Variant tag for the DNS4 16 kHz enhancement checkpoint.
pub const VARIANT_TAG_DNS4_ENHANCEMENT: &str = "dns4-16k-enhancement";

const KEY_MODEL_ID: &str = "vokra.provenance.model_id";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const CHANNELS: usize = 256;
const FFN: usize = 1024;
const HEADS: usize = 8;
const TRANSFORMER_LAYERS: usize = 8;
const DUAL_BLOCKS: usize = 2;
const CHUNK: usize = 250;
const CHUNK_HOP: usize = CHUNK / 2;
const ENCODER_KERNEL: usize = 16;
const ENCODER_STRIDE: usize = 8;
const POSITION_LIMIT: usize = 2500;
const TRANSFORMER_EPS: f32 = 1e-6;
const GROUP_NORM_EPS: f32 = 1e-8;
const TENSOR_COUNT: usize = 417;

/// Complete learned-op registry for native CPU/Metal SepFormer inference.
pub const SEPFORMER_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::GroupNorm,
    HotOp::Conv1d,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Supported public SpeechBrain SepFormer checkpoints.
pub enum SepformerVariant {
    /// WSJ0 2-speaker source separation at 8 kHz.
    Wsj02mix,
    /// LibriMix 2-speaker source separation at 8 kHz.
    Libri2Mix,
    /// LibriMix 3-speaker source separation at 8 kHz.
    Libri3Mix,
    /// WHAM! single-stream enhancement at 16 kHz.
    Wham16kEnhancement,
    /// WHAMR! 2-speaker source separation at 16 kHz.
    Whamr16k,
    /// WHAMR! 2-speaker source separation at 8 kHz.
    Whamr8k,
    /// DNS4 single-stream enhancement at 16 kHz.
    Dns4Enhancement,
}

impl SepformerVariant {
    /// All supported SepFormer variants in stable catalog order.
    pub const ALL: [Self; 7] = [
        Self::Wsj02mix,
        Self::Libri2Mix,
        Self::Libri3Mix,
        Self::Wham16kEnhancement,
        Self::Whamr16k,
        Self::Whamr8k,
        Self::Dns4Enhancement,
    ];

    /// Returns the canonical Vokra model identifier.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Wsj02mix => "sepformer-wsj02mix",
            Self::Libri2Mix => "sepformer-libri2mix",
            Self::Libri3Mix => "sepformer-libri3mix",
            Self::Wham16kEnhancement => "sepformer-wham16k-enhancement",
            Self::Whamr16k => "sepformer-whamr16k",
            Self::Whamr8k => "sepformer-whamr",
            Self::Dns4Enhancement => "sepformer-dns4-16k-enhancement",
        }
    }

    /// Returns the pinned upstream Hugging Face repository identifier.
    #[must_use]
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Wsj02mix => "speechbrain/sepformer-wsj02mix",
            Self::Libri2Mix => "speechbrain/sepformer-libri2mix",
            Self::Libri3Mix => "speechbrain/sepformer-libri3mix",
            Self::Wham16kEnhancement => "speechbrain/sepformer-wham16k-enhancement",
            Self::Whamr16k => "speechbrain/sepformer-whamr16k",
            Self::Whamr8k => "speechbrain/sepformer-whamr",
            Self::Dns4Enhancement => "speechbrain/sepformer-dns4-16k-enhancement",
        }
    }

    /// Returns the value stored in [`KEY_SEPFORMER_VARIANT`].
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Wsj02mix => VARIANT_TAG_WSJ02MIX,
            Self::Libri2Mix => VARIANT_TAG_LIBRI2MIX,
            Self::Libri3Mix => VARIANT_TAG_LIBRI3MIX,
            Self::Wham16kEnhancement => VARIANT_TAG_WHAM16K_ENHANCEMENT,
            Self::Whamr16k => VARIANT_TAG_WHAMR16K,
            Self::Whamr8k => VARIANT_TAG_WHAMR8K,
            Self::Dns4Enhancement => VARIANT_TAG_DNS4_ENHANCEMENT,
        }
    }

    /// Returns the model category metadata value.
    #[must_use]
    pub const fn category(self) -> &'static str {
        match self {
            Self::Wsj02mix | Self::Libri2Mix | Self::Libri3Mix | Self::Whamr16k | Self::Whamr8k => {
                CATEGORY_SEPARATION
            }
            Self::Wham16kEnhancement | Self::Dns4Enhancement => CATEGORY_ENHANCEMENT,
        }
    }

    /// Returns the number of output waveform streams.
    #[must_use]
    pub const fn n_out(self) -> u32 {
        match self {
            Self::Wham16kEnhancement | Self::Dns4Enhancement => 1,
            Self::Wsj02mix | Self::Libri2Mix | Self::Whamr16k | Self::Whamr8k => 2,
            Self::Libri3Mix => 3,
        }
    }

    /// Returns the required input and output sample rate in hertz.
    #[must_use]
    pub const fn sample_rate(self) -> u32 {
        match self {
            Self::Wsj02mix | Self::Libri2Mix | Self::Libri3Mix | Self::Whamr8k => 8_000,
            _ => 16_000,
        }
    }

    /// Parses an exact [`KEY_SEPFORMER_VARIANT`] metadata value.
    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            VARIANT_TAG_WSJ02MIX => Some(Self::Wsj02mix),
            VARIANT_TAG_LIBRI2MIX => Some(Self::Libri2Mix),
            VARIANT_TAG_LIBRI3MIX => Some(Self::Libri3Mix),
            VARIANT_TAG_WHAM16K_ENHANCEMENT => Some(Self::Wham16kEnhancement),
            VARIANT_TAG_WHAMR16K => Some(Self::Whamr16k),
            VARIANT_TAG_WHAMR8K => Some(Self::Whamr8k),
            VARIANT_TAG_DNS4_ENHANCEMENT => Some(Self::Dns4Enhancement),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Static topology and audio configuration for one SepFormer checkpoint.
pub struct SepformerConfig {
    /// Concrete public checkpoint variant.
    pub variant: SepformerVariant,
    /// Number of separated or enhanced output streams.
    pub n_out: u32,
    /// Model category metadata value.
    pub category: &'static str,
    /// Required waveform sample rate in hertz.
    pub sample_rate: u32,
}

impl SepformerConfig {
    /// Builds the fixed configuration for `variant`.
    #[must_use]
    pub const fn for_variant(variant: SepformerVariant) -> Self {
        Self {
            variant,
            n_out: variant.n_out(),
            category: variant.category(),
            sample_rate: variant.sample_rate(),
        }
    }
}

#[derive(Debug)]
struct Linear {
    weight_t: Vec<f32>,
    bias: Option<Vec<f32>>,
    input: usize,
    output: usize,
}

impl Linear {
    fn bind(
        file: &GgufFile,
        weight_name: &str,
        bias_name: Option<&str>,
        input: usize,
        output: usize,
        trailing_ones: usize,
    ) -> Result<Self> {
        let mut dims = vec![output, input];
        dims.extend(std::iter::repeat_n(1, trailing_ones));
        let weight = tensor(file, weight_name, &dims)?;
        let mut weight_t = vec![0.0f32; weight.len()];
        for out in 0..output {
            for input_index in 0..input {
                weight_t[input_index * output + out] = weight[out * input + input_index];
            }
        }
        let bias = bias_name
            .map(|name| tensor(file, name, &[output]))
            .transpose()?;
        Ok(Self {
            weight_t,
            bias,
            input,
            output,
        })
    }

    fn forward(&self, values: &[f32], rows: usize, compute: &Compute) -> Result<Vec<f32>> {
        if values.len() != rows * self.input {
            return Err(VokraError::InvalidArgument(format!(
                "sepformer: linear input has {} values, expected {}x{}",
                values.len(),
                rows,
                self.input
            )));
        }
        let mut output = vec![0.0f32; rows * self.output];
        compute.gemm_f32(
            rows,
            self.output,
            self.input,
            values,
            &self.weight_t,
            self.bias.as_deref(),
            &mut output,
        )?;
        Ok(output)
    }
}

#[derive(Debug)]
struct TransformerLayer {
    norm1_gamma: Vec<f32>,
    norm1_beta: Vec<f32>,
    norm2_gamma: Vec<f32>,
    norm2_beta: Vec<f32>,
    qkv: Linear,
    out: Linear,
    ff1: Linear,
    ff2: Linear,
}

#[derive(Debug)]
struct TransformerStack {
    layers: Vec<TransformerLayer>,
    final_gamma: Vec<f32>,
    final_beta: Vec<f32>,
    position: Vec<f32>,
}

impl TransformerStack {
    fn bind(file: &GgufFile, prefix: &str) -> Result<Self> {
        let mut layers = Vec::with_capacity(TRANSFORMER_LAYERS);
        for layer in 0..TRANSFORMER_LAYERS {
            let stem = format!("{prefix}.mdl.layers.{layer}");
            layers.push(TransformerLayer {
                norm1_gamma: tensor(file, &format!("{stem}.norm1.norm.weight"), &[CHANNELS])?,
                norm1_beta: tensor(file, &format!("{stem}.norm1.norm.bias"), &[CHANNELS])?,
                norm2_gamma: tensor(file, &format!("{stem}.norm2.norm.weight"), &[CHANNELS])?,
                norm2_beta: tensor(file, &format!("{stem}.norm2.norm.bias"), &[CHANNELS])?,
                qkv: Linear::bind(
                    file,
                    &format!("{stem}.self_att.att.in_proj_weight"),
                    Some(&format!("{stem}.self_att.att.in_proj_bias")),
                    CHANNELS,
                    CHANNELS * 3,
                    0,
                )?,
                out: Linear::bind(
                    file,
                    &format!("{stem}.self_att.att.out_proj.weight"),
                    Some(&format!("{stem}.self_att.att.out_proj.bias")),
                    CHANNELS,
                    CHANNELS,
                    0,
                )?,
                ff1: Linear::bind(
                    file,
                    &format!("{stem}.pos_ffn.ffn.0.weight"),
                    Some(&format!("{stem}.pos_ffn.ffn.0.bias")),
                    CHANNELS,
                    FFN,
                    0,
                )?,
                ff2: Linear::bind(
                    file,
                    &format!("{stem}.pos_ffn.ffn.3.weight"),
                    Some(&format!("{stem}.pos_ffn.ffn.3.bias")),
                    FFN,
                    CHANNELS,
                    0,
                )?,
            });
        }
        Ok(Self {
            layers,
            final_gamma: tensor(file, &format!("{prefix}.mdl.norm.norm.weight"), &[CHANNELS])?,
            final_beta: tensor(file, &format!("{prefix}.mdl.norm.norm.bias"), &[CHANNELS])?,
            position: tensor(
                file,
                &format!("{prefix}.pos_enc.pe"),
                &[1, POSITION_LIMIT, CHANNELS],
            )?,
        })
    }

    fn forward(
        &self,
        values: &mut [f32],
        batches: usize,
        sequence: usize,
        compute: &Compute,
    ) -> Result<()> {
        if sequence > POSITION_LIMIT {
            return Err(VokraError::InvalidArgument(format!(
                "sepformer: Transformer sequence length {sequence} exceeds positional limit {POSITION_LIMIT}"
            )));
        }
        let rows = batches * sequence;
        for batch in 0..batches {
            for position in 0..sequence {
                let destination = (batch * sequence + position) * CHANNELS;
                let source = position * CHANNELS;
                for channel in 0..CHANNELS {
                    values[destination + channel] += self.position[source + channel];
                }
            }
        }
        for layer in &self.layers {
            let normalized = layer_norm(
                values,
                rows,
                CHANNELS,
                &layer.norm1_gamma,
                &layer.norm1_beta,
                TRANSFORMER_EPS,
                compute,
            )?;
            let qkv = layer.qkv.forward(&normalized, rows, compute)?;
            let attention = attention(&qkv, batches, sequence, compute)?;
            let projected = layer.out.forward(&attention, rows, compute)?;
            add_inplace(values, &projected);

            let normalized = layer_norm(
                values,
                rows,
                CHANNELS,
                &layer.norm2_gamma,
                &layer.norm2_beta,
                TRANSFORMER_EPS,
                compute,
            )?;
            let mut hidden = layer.ff1.forward(&normalized, rows, compute)?;
            for value in &mut hidden {
                *value = value.max(0.0);
            }
            let projected = layer.ff2.forward(&hidden, rows, compute)?;
            add_inplace(values, &projected);
        }
        let normalized = layer_norm(
            values,
            rows,
            CHANNELS,
            &self.final_gamma,
            &self.final_beta,
            TRANSFORMER_EPS,
            compute,
        )?;
        values.copy_from_slice(&normalized);
        Ok(())
    }
}

#[derive(Debug)]
struct DualBlock {
    intra: TransformerStack,
    inter: TransformerStack,
    intra_gamma: Vec<f32>,
    intra_beta: Vec<f32>,
    inter_gamma: Vec<f32>,
    inter_beta: Vec<f32>,
}

impl DualBlock {
    fn bind(file: &GgufFile, index: usize) -> Result<Self> {
        let prefix = format!("masknet.dual_mdl.{index}");
        Ok(Self {
            intra: TransformerStack::bind(file, &format!("{prefix}.intra_mdl"))?,
            inter: TransformerStack::bind(file, &format!("{prefix}.inter_mdl"))?,
            intra_gamma: tensor(file, &format!("{prefix}.intra_norm.weight"), &[CHANNELS])?,
            intra_beta: tensor(file, &format!("{prefix}.intra_norm.bias"), &[CHANNELS])?,
            inter_gamma: tensor(file, &format!("{prefix}.inter_norm.weight"), &[CHANNELS])?,
            inter_beta: tensor(file, &format!("{prefix}.inter_norm.bias"), &[CHANNELS])?,
        })
    }

    fn forward(&self, values: &mut [f32], sequence_count: usize, compute: &Compute) -> Result<()> {
        let residual = values.to_vec();
        let mut intra = dual_to_intra_rows(values, sequence_count);
        self.intra
            .forward(&mut intra, sequence_count, CHUNK, compute)?;
        let intra = intra_rows_to_dual(&intra, sequence_count);
        values.copy_from_slice(&intra);
        group_norm(
            values,
            CHUNK * sequence_count,
            &self.intra_gamma,
            &self.intra_beta,
            compute,
        )?;
        add_inplace(values, &residual);

        let mut inter = dual_to_inter_rows(values, sequence_count);
        self.inter
            .forward(&mut inter, CHUNK, sequence_count, compute)?;
        let mut inter = inter_rows_to_dual(&inter, sequence_count);
        group_norm(
            &mut inter,
            CHUNK * sequence_count,
            &self.inter_gamma,
            &self.inter_beta,
            compute,
        )?;
        add_inplace(values, &inter);
        Ok(())
    }
}

#[derive(Debug)]
struct NetworkWeights {
    encoder: Vec<f32>,
    decoder_matrix: Vec<f32>,
    mask_norm_gamma: Vec<f32>,
    mask_norm_beta: Vec<f32>,
    mask_input: Vec<f32>,
    dual: Vec<DualBlock>,
    prelu: f32,
    speaker_projection: Linear,
    output: Linear,
    output_gate: Linear,
    end: Linear,
}

impl NetworkWeights {
    fn bind(file: &GgufFile, n_out: usize) -> Result<Self> {
        let prelu = tensor(file, "masknet.prelu.weight", &[1])?[0];
        Ok(Self {
            encoder: tensor(
                file,
                "encoder.conv1d.weight",
                &[CHANNELS, 1, ENCODER_KERNEL],
            )?,
            decoder_matrix: tensor(file, "decoder.weight", &[CHANNELS, 1, ENCODER_KERNEL])?,
            mask_norm_gamma: tensor(file, "masknet.norm.weight", &[CHANNELS])?,
            mask_norm_beta: tensor(file, "masknet.norm.bias", &[CHANNELS])?,
            mask_input: tensor(file, "masknet.conv1d.weight", &[CHANNELS, CHANNELS, 1])?,
            dual: (0..DUAL_BLOCKS)
                .map(|index| DualBlock::bind(file, index))
                .collect::<Result<Vec<_>>>()?,
            prelu,
            speaker_projection: Linear::bind(
                file,
                "masknet.conv2d.weight",
                Some("masknet.conv2d.bias"),
                CHANNELS,
                CHANNELS * n_out,
                2,
            )?,
            output: Linear::bind(
                file,
                "masknet.output.0.weight",
                Some("masknet.output.0.bias"),
                CHANNELS,
                CHANNELS,
                1,
            )?,
            output_gate: Linear::bind(
                file,
                "masknet.output_gate.0.weight",
                Some("masknet.output_gate.0.bias"),
                CHANNELS,
                CHANNELS,
                1,
            )?,
            end: Linear::bind(
                file,
                "masknet.end_conv1x1.weight",
                None,
                CHANNELS,
                CHANNELS,
                1,
            )?,
        })
    }
}

#[derive(Debug)]
/// Strictly bound learned SepFormer tensors.
pub struct SepformerWeights {
    network: Option<Box<NetworkWeights>>,
    tensor_count: usize,
}

impl SepformerWeights {
    /// Binds and validates every required tensor in `file`.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let tag = required_string(file, KEY_SEPFORMER_VARIANT)?;
        let variant = SepformerVariant::from_tag(tag).ok_or_else(|| {
            VokraError::ModelLoad(format!("sepformer: unsupported variant tag {tag:?}"))
        })?;
        let n_out = audited_n_out(file, variant)? as usize;
        if file.tensors().len() != TENSOR_COUNT {
            return Err(VokraError::ModelLoad(format!(
                "sepformer: GGUF has {} tensors, expected exactly {TENSOR_COUNT}",
                file.tensors().len()
            )));
        }
        Ok(Self {
            network: Some(Box::new(NetworkWeights::bind(file, n_out)?)),
            tensor_count: file.tensors().len(),
        })
    }

    /// Returns the exact number of tensors accepted from the GGUF.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensor_count
    }
}

#[derive(Debug)]
/// Native SepFormer inference handle with explicit CPU or Metal dispatch.
pub struct SepFormer {
    config: SepformerConfig,
    weights: SepformerWeights,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl SepFormer {
    /// Strictly binds a complete public SepFormer GGUF.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        let tag = required_string(file, KEY_SEPFORMER_VARIANT)?;
        let variant = SepformerVariant::from_tag(tag).ok_or_else(|| {
            VokraError::ModelLoad(format!("sepformer: unsupported variant tag {tag:?}"))
        })?;
        require_string(file, chunks::KEY_MODEL_NAME, variant.name())?;
        require_string(file, KEY_MODEL_ID, variant.name())?;
        audited_category(file, variant)?;
        require_string(file, KEY_UPSTREAM_HF, variant.upstream_hf())?;
        audited_n_out(file, variant)?;
        let weights = SepformerWeights::from_gguf(file)?;
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|value| value.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        Ok(Self {
            config: SepformerConfig::for_variant(variant),
            weights,
            weight_license,
            backend: BackendKind::Cpu,
        })
    }

    /// Selects the execution backend without adding a fallback path.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    fn compute(&self) -> Result<Compute> {
        if self.config.variant == SepformerVariant::Dns4Enhancement {
            // DNS4's mask network is unusually ill-conditioned: the NEON FMA
            // reduction order amplifies an otherwise sub-ulp encoder delta
            // past the independently measured FP64 waveform boundary. Keep
            // the portable reduction order for this CPU variant only. Metal
            // remains a fully GPU-dispatched backend with no CPU fallback.
            Compute::for_backend_with_scalar_cpu(self.backend, SEPFORMER_HOT_OPS)
        } else {
            Compute::for_backend(self.backend, SEPFORMER_HOT_OPS)
        }
    }

    /// Returns the selected execution backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the checkpoint configuration.
    #[must_use]
    pub const fn config(&self) -> &SepformerConfig {
        &self.config
    }

    /// Returns the concrete checkpoint variant.
    #[must_use]
    pub const fn variant(&self) -> SepformerVariant {
        self.config.variant
    }

    /// Returns the number of output waveform streams.
    #[must_use]
    pub const fn n_out(&self) -> u32 {
        self.config.n_out
    }

    /// Returns the model category metadata value.
    #[must_use]
    pub const fn category(&self) -> &'static str {
        self.config.category
    }

    /// Returns the required waveform sample rate in hertz.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Returns the exact number of bound GGUF tensors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Returns the normalized license class stamped in provenance metadata.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Creates a metadata-only handle used to verify explicit missing-weight errors.
    #[must_use]
    pub fn synthesized(variant: SepformerVariant) -> Self {
        Self {
            config: SepformerConfig::for_variant(variant),
            weights: SepformerWeights {
                network: None,
                tensor_count: 1,
            },
            weight_license: LicenseClass::Unknown,
            backend: BackendKind::Cpu,
        }
    }

    /// Separates or enhances a mono waveform using the bound checkpoint.
    pub fn separate(&self, mixed_pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        if mixed_pcm.len() < ENCODER_KERNEL {
            return Err(VokraError::InvalidArgument(format!(
                "sepformer: input has {} samples, minimum is {ENCODER_KERNEL}",
                mixed_pcm.len()
            )));
        }
        let weights = self.weights.network.as_deref().ok_or_else(|| {
            VokraError::UnsupportedOp(
                "sepformer: synthesized test handle has no learned weights".to_owned(),
            )
        })?;
        let compute = self.compute()?;
        separate_network(mixed_pcm, self.config.n_out as usize, weights, &compute)
    }

    /// Runs the learned waveform encoder and returns channel-major
    /// `[256, frames]` features plus the frame count.
    pub fn encode_features(&self, mixed_pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        if mixed_pcm.len() < ENCODER_KERNEL {
            return Err(VokraError::InvalidArgument(format!(
                "sepformer: input has {} samples, minimum is {ENCODER_KERNEL}",
                mixed_pcm.len()
            )));
        }
        let weights = self.weights.network.as_deref().ok_or_else(|| {
            VokraError::UnsupportedOp(
                "sepformer: synthesized test handle has no learned weights".to_owned(),
            )
        })?;
        let compute = self.compute()?;
        encode_network(mixed_pcm, &weights.encoder, &compute)
    }
}

impl vokra_core::engines::SeparationEngine for SepFormer {
    fn separate(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        SepFormer::separate(self, pcm)
    }

    fn sample_rate(&self) -> u32 {
        SepFormer::sample_rate(self)
    }

    fn output_streams(&self) -> usize {
        self.n_out() as usize
    }

    fn backend(&self) -> BackendKind {
        SepFormer::backend(self)
    }
}

fn separate_network(
    pcm: &[f32],
    n_out: usize,
    weights: &NetworkWeights,
    compute: &Compute,
) -> Result<Vec<Vec<f32>>> {
    let (encoded, frames) = encode_network(pcm, &weights.encoder, compute)?;
    let mut masker = encoded.clone();
    group_norm(
        &mut masker,
        frames,
        &weights.mask_norm_gamma,
        &weights.mask_norm_beta,
        compute,
    )?;
    let mut projected = vec![0.0f32; masker.len()];
    compute.conv1d_f32(
        &masker,
        CHANNELS,
        frames,
        &weights.mask_input,
        CHANNELS,
        1,
        None,
        1,
        0,
        &mut projected,
    )?;

    let (mut chunked, sequence_count, gap) = segment(&projected, frames);
    for block in &weights.dual {
        block.forward(&mut chunked, sequence_count, compute)?;
    }
    for value in &mut chunked {
        *value = if *value >= 0.0 {
            *value
        } else {
            *value * weights.prelu
        };
    }

    let positions = CHUNK * sequence_count;
    let rows = dual_to_position_rows(&chunked, sequence_count);
    let speaker_rows = weights
        .speaker_projection
        .forward(&rows, positions, compute)?;
    let speaker_chunks = speaker_rows_to_dual(&speaker_rows, n_out, sequence_count);
    let mut masks = Vec::with_capacity(n_out);
    for speaker in 0..n_out {
        let begin = speaker * CHANNELS * positions;
        let end = begin + CHANNELS * positions;
        masks.push(overlap_add(
            &speaker_chunks[begin..end],
            sequence_count,
            gap,
        ));
    }

    let mut outputs = Vec::with_capacity(n_out);
    for mask in &mut masks {
        let rows = channel_to_frame(mask, frames);
        let output = weights.output.forward(&rows, frames, compute)?;
        let gate = weights.output_gate.forward(&rows, frames, compute)?;
        let gated = output
            .iter()
            .zip(gate)
            .map(|(&value, gate)| value.tanh() * sigmoid(gate))
            .collect::<Vec<_>>();
        let mut mask_rows = weights.end.forward(&gated, frames, compute)?;
        for value in &mut mask_rows {
            *value = value.max(0.0);
        }
        *mask = frame_to_channel(&mask_rows, frames);

        let mut separated = vec![0.0f32; encoded.len()];
        for index in 0..separated.len() {
            separated[index] = encoded[index] * mask[index];
        }
        outputs.push(decode(
            &separated,
            frames,
            pcm.len(),
            &weights.decoder_matrix,
            compute,
        )?);
    }
    Ok(outputs)
}

fn encode_network(pcm: &[f32], encoder: &[f32], compute: &Compute) -> Result<(Vec<f32>, usize)> {
    let frames = (pcm.len() - ENCODER_KERNEL) / ENCODER_STRIDE + 1;
    let mut encoded = vec![0.0f32; CHANNELS * frames];
    compute.conv1d_f32(
        pcm,
        1,
        pcm.len(),
        encoder,
        CHANNELS,
        ENCODER_KERNEL,
        None,
        ENCODER_STRIDE,
        0,
        &mut encoded,
    )?;
    for value in &mut encoded {
        *value = value.max(0.0);
    }
    Ok((encoded, frames))
}

fn attention(qkv: &[f32], batches: usize, sequence: usize, compute: &Compute) -> Result<Vec<f32>> {
    let head_dim = CHANNELS / HEADS;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; batches * sequence * CHANNELS];
    let mut q = vec![0.0f32; sequence * head_dim];
    let mut k_t = vec![0.0f32; head_dim * sequence];
    let mut v = vec![0.0f32; sequence * head_dim];
    let mut scores = vec![0.0f32; sequence * sequence];
    let mut probabilities = vec![0.0f32; sequence * sequence];
    let mut attended = vec![0.0f32; sequence * head_dim];
    for batch in 0..batches {
        for head in 0..HEADS {
            for position in 0..sequence {
                let source = (batch * sequence + position) * CHANNELS * 3;
                for dim in 0..head_dim {
                    let channel = head * head_dim + dim;
                    q[position * head_dim + dim] = qkv[source + channel];
                    k_t[dim * sequence + position] = qkv[source + CHANNELS + channel];
                    v[position * head_dim + dim] = qkv[source + CHANNELS * 2 + channel];
                }
            }
            compute.gemm_f32(sequence, sequence, head_dim, &q, &k_t, None, &mut scores)?;
            for score in &mut scores {
                *score *= scale;
            }
            compute.softmax_f32(&scores, &mut probabilities, sequence, sequence)?;
            compute.gemm_f32(
                sequence,
                head_dim,
                sequence,
                &probabilities,
                &v,
                None,
                &mut attended,
            )?;
            for position in 0..sequence {
                let destination = (batch * sequence + position) * CHANNELS + head * head_dim;
                output[destination..destination + head_dim]
                    .copy_from_slice(&attended[position * head_dim..(position + 1) * head_dim]);
            }
        }
    }
    Ok(output)
}

fn layer_norm(
    input: &[f32],
    rows: usize,
    columns: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    compute: &Compute,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; input.len()];
    compute.layer_norm_f32(input, &mut output, rows, columns, gamma, beta, eps)?;
    Ok(output)
}

fn group_norm(
    values: &mut [f32],
    positions: usize,
    gamma: &[f32],
    beta: &[f32],
    compute: &Compute,
) -> Result<()> {
    let mut normalized = vec![0.0f32; values.len()];
    compute.group_norm_f32(
        values,
        &mut normalized,
        CHANNELS,
        positions,
        gamma,
        beta,
        GROUP_NORM_EPS,
    )?;
    values.copy_from_slice(&normalized);
    Ok(())
}

fn segment(input: &[f32], frames: usize) -> (Box<[f32]>, usize, usize) {
    let gap = CHUNK - (CHUNK_HOP + frames % CHUNK) % CHUNK;
    let padded_frames = frames + gap + CHUNK;
    let chunk_pairs = (padded_frames - CHUNK_HOP) / CHUNK;
    let sequence_count = chunk_pairs * 2;
    let mut output = vec![0.0f32; CHANNELS * CHUNK * sequence_count];
    for channel in 0..CHANNELS {
        for pair in 0..chunk_pairs {
            for within in 0..CHUNK {
                let first = pair * CHUNK + within;
                let second = first + CHUNK_HOP;
                let destination = (channel * CHUNK + within) * sequence_count + pair * 2;
                output[destination] = padded_value(input, frames, channel, first);
                output[destination + 1] = padded_value(input, frames, channel, second);
            }
        }
    }
    (output.into_boxed_slice(), sequence_count, gap)
}

fn padded_value(input: &[f32], frames: usize, channel: usize, padded_index: usize) -> f32 {
    if padded_index < CHUNK_HOP || padded_index >= CHUNK_HOP + frames {
        0.0
    } else {
        input[channel * frames + padded_index - CHUNK_HOP]
    }
}

fn overlap_add(input: &[f32], sequence_count: usize, gap: usize) -> Vec<f32> {
    let pairs = sequence_count / 2;
    let joined = pairs * CHUNK;
    let output_frames = joined - CHUNK_HOP - gap;
    let mut output = vec![0.0f32; CHANNELS * output_frames];
    for channel in 0..CHANNELS {
        for frame in 0..output_frames {
            let first_index = frame + CHUNK_HOP;
            let second_index = frame;
            let first_pair = first_index / CHUNK;
            let first_within = first_index % CHUNK;
            let second_pair = second_index / CHUNK;
            let second_within = second_index % CHUNK;
            let first = input[(channel * CHUNK + first_within) * sequence_count + first_pair * 2];
            let second =
                input[(channel * CHUNK + second_within) * sequence_count + second_pair * 2 + 1];
            output[channel * output_frames + frame] = first + second;
        }
    }
    output
}

fn decode(
    latent: &[f32],
    frames: usize,
    original_samples: usize,
    decoder_matrix: &[f32],
    compute: &Compute,
) -> Result<Vec<f32>> {
    let frame_rows = channel_to_frame(latent, frames);
    let mut kernels = vec![0.0f32; frames * ENCODER_KERNEL];
    compute.gemm_f32(
        frames,
        ENCODER_KERNEL,
        CHANNELS,
        &frame_rows,
        decoder_matrix,
        None,
        &mut kernels,
    )?;
    let decoded_len = (frames - 1) * ENCODER_STRIDE + ENCODER_KERNEL;
    let mut decoded = vec![0.0f32; decoded_len.max(original_samples)];
    for frame in 0..frames {
        for kernel in 0..ENCODER_KERNEL {
            decoded[frame * ENCODER_STRIDE + kernel] += kernels[frame * ENCODER_KERNEL + kernel];
        }
    }
    decoded.resize(original_samples, 0.0);
    Ok(decoded)
}

fn dual_to_intra_rows(input: &[f32], sequence_count: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for sequence in 0..sequence_count {
        for within in 0..CHUNK {
            for channel in 0..CHANNELS {
                output[(sequence * CHUNK + within) * CHANNELS + channel] =
                    input[(channel * CHUNK + within) * sequence_count + sequence];
            }
        }
    }
    output
}

fn intra_rows_to_dual(input: &[f32], sequence_count: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for sequence in 0..sequence_count {
        for within in 0..CHUNK {
            for channel in 0..CHANNELS {
                output[(channel * CHUNK + within) * sequence_count + sequence] =
                    input[(sequence * CHUNK + within) * CHANNELS + channel];
            }
        }
    }
    output
}

fn dual_to_inter_rows(input: &[f32], sequence_count: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for within in 0..CHUNK {
        for sequence in 0..sequence_count {
            for channel in 0..CHANNELS {
                output[(within * sequence_count + sequence) * CHANNELS + channel] =
                    input[(channel * CHUNK + within) * sequence_count + sequence];
            }
        }
    }
    output
}

fn inter_rows_to_dual(input: &[f32], sequence_count: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for within in 0..CHUNK {
        for sequence in 0..sequence_count {
            for channel in 0..CHANNELS {
                output[(channel * CHUNK + within) * sequence_count + sequence] =
                    input[(within * sequence_count + sequence) * CHANNELS + channel];
            }
        }
    }
    output
}

fn dual_to_position_rows(input: &[f32], sequence_count: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for within in 0..CHUNK {
        for sequence in 0..sequence_count {
            for channel in 0..CHANNELS {
                output[(within * sequence_count + sequence) * CHANNELS + channel] =
                    input[(channel * CHUNK + within) * sequence_count + sequence];
            }
        }
    }
    output
}

fn speaker_rows_to_dual(input: &[f32], n_out: usize, sequence_count: usize) -> Vec<f32> {
    let positions = CHUNK * sequence_count;
    let mut output = vec![0.0f32; n_out * CHANNELS * positions];
    for within in 0..CHUNK {
        for sequence in 0..sequence_count {
            let row = within * sequence_count + sequence;
            for speaker in 0..n_out {
                for channel in 0..CHANNELS {
                    output[((speaker * CHANNELS + channel) * CHUNK + within) * sequence_count
                        + sequence] = input[row * n_out * CHANNELS + speaker * CHANNELS + channel];
                }
            }
        }
    }
    output
}

fn channel_to_frame(input: &[f32], frames: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for frame in 0..frames {
        for channel in 0..CHANNELS {
            output[frame * CHANNELS + channel] = input[channel * frames + frame];
        }
    }
    output
}

fn frame_to_channel(input: &[f32], frames: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for frame in 0..frames {
        for channel in 0..CHANNELS {
            output[channel * frames + frame] = input[frame * CHANNELS + channel];
        }
    }
    output
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn add_inplace(destination: &mut [f32], source: &[f32]) {
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += source;
    }
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("sepformer: missing tensor `{name}`")))?;
    let expected_u64 = expected
        .iter()
        .map(|&value| value as u64)
        .collect::<Vec<_>>();
    if info.dimensions != expected_u64 {
        return Err(VokraError::ModelLoad(format!(
            "sepformer: tensor `{name}` has dims {:?}, expected {expected_u64:?}",
            info.dimensions
        )));
    }
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("sepformer: reading tensor `{name}`: {error}"))
    })
}

fn required_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| VokraError::ModelLoad(format!("sepformer: missing string `{key}`")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = required_string(file, key)?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "sepformer: `{key}` is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn audited_n_out(file: &GgufFile, variant: SepformerVariant) -> Result<u64> {
    let expected = u64::from(variant.n_out());
    match file.get(KEY_SEPFORMER_N_OUT) {
        Some(value) => {
            let actual = value.as_u64().ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "sepformer: `{KEY_SEPFORMER_N_OUT}` is not an unsigned integer"
                ))
            })?;
            if actual == expected {
                Ok(expected)
            } else if actual == 1
                && matches!(
                    variant,
                    SepformerVariant::Whamr16k | SepformerVariant::Whamr8k
                )
            {
                // Both first public WHAMR artifacts were stamped as
                // single-output enhancement even though the official
                // checkpoints and [512, 256, 1, 1] projection contain two
                // separated speakers. Exact variant/provenance checks plus
                // strict tensor shapes bound this repair to those releases.
                Ok(expected)
            } else {
                Err(VokraError::ModelLoad(format!(
                    "sepformer: {KEY_SEPFORMER_N_OUT}={actual}, expected {expected} for {}",
                    variant.tag()
                )))
            }
        }
        None if variant == SepformerVariant::Wsj02mix => {
            // The first public vokra/sepformer-wsj02mix artifact predates the
            // additive n_out stamp. The strict variant/provenance checks and
            // the [512, 256, 1, 1] speaker projection below independently pin
            // the only accepted legacy value to two streams.
            Ok(2)
        }
        None => Err(VokraError::ModelLoad(format!(
            "sepformer: missing unsigned `{KEY_SEPFORMER_N_OUT}`"
        ))),
    }
}

fn audited_category(file: &GgufFile, variant: SepformerVariant) -> Result<()> {
    let actual = required_string(file, KEY_MODEL_CATEGORY)?;
    let expected = variant.category();
    if actual == expected
        || (actual == CATEGORY_ENHANCEMENT
            && matches!(
                variant,
                SepformerVariant::Whamr16k | SepformerVariant::Whamr8k
            ))
    {
        Ok(())
    } else {
        Err(VokraError::ModelLoad(format!(
            "sepformer: `{KEY_MODEL_CATEGORY}` is {actual:?}, expected {expected:?}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_pin_public_contract() {
        let rows = [
            (SepformerVariant::Wsj02mix, "sepformer-wsj02mix", 2, 8_000),
            (SepformerVariant::Libri2Mix, "sepformer-libri2mix", 2, 8_000),
            (SepformerVariant::Libri3Mix, "sepformer-libri3mix", 3, 8_000),
            (
                SepformerVariant::Wham16kEnhancement,
                "sepformer-wham16k-enhancement",
                1,
                16_000,
            ),
            (SepformerVariant::Whamr16k, "sepformer-whamr16k", 2, 16_000),
            (SepformerVariant::Whamr8k, "sepformer-whamr", 2, 8_000),
            (
                SepformerVariant::Dns4Enhancement,
                "sepformer-dns4-16k-enhancement",
                1,
                16_000,
            ),
        ];
        for (variant, name, n_out, sample_rate) in rows {
            assert_eq!(variant.name(), name);
            assert_eq!(variant.n_out(), n_out);
            assert_eq!(variant.sample_rate(), sample_rate);
            assert_eq!(SepformerVariant::from_tag(variant.tag()), Some(variant));
        }
    }

    #[test]
    fn segmentation_and_overlap_add_are_inverse_for_ones() {
        let frames = 511;
        let input = vec![1.0f32; CHANNELS * frames];
        let (chunks, sequences, gap) = segment(&input, frames);
        let restored = overlap_add(&chunks, sequences, gap);
        assert_eq!(restored.len(), input.len());
        assert!(restored.iter().all(|&value| value == 2.0));
    }

    #[test]
    fn synthesized_handle_stays_explicit() {
        let model = SepFormer::synthesized(SepformerVariant::Wham16kEnhancement);
        let error = model.separate(&[0.0; 32]).unwrap_err();
        assert!(matches!(error, VokraError::UnsupportedOp(_)));
    }
}
