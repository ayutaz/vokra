//! Native WavTokenizer large-speech 75 token/s decoder.
//!
//! The two public `vokra/wavtokenizer-*` repositories contain the same
//! 846,393,344-byte GGUF (SHA-256
//! `99b7dce0426266f7f2f6615091d832cea71387ce57edfae66666143a5c33a36b`).
//! This binder recognizes the released 24 kHz / hop-320 / one-codebook
//! topology, validates every inference tensor, and ignores only explicitly
//! training-only checkpoint families (discriminators, losses and the unused
//! Encodec waveform decoder).
//!
//! Decode follows the pinned upstream order exactly:
//! codebook gather -> 512-to-768 Conv1D -> four GroupNorm/SiLU residual
//! blocks around one full attention block -> AdaLayerNorm ConvNeXt stack ->
//! magnitude/phase iSTFT. Every learned operation and SiLU is dispatched via
//! [`Compute`]. Unsupported backends fail before inference; there is no CPU
//! fallback.

use std::collections::BTreeSet;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::vocos::vocos_decode_from_embedded_with_ops;
use vokra_ops::{
    CodebookTable, VocosAttrs, VocosBackendOps, VocosBlockWeights, VocosIstftPadding,
    VocosNormWeights, VocosWeights, WavTokenizerVqAttrs,
};

use crate::compute::{Compute, HotOp};

/// Converter/runtime architecture tag.
pub const ARCH: &str = "wavtokenizer";
/// Exact released public model id.
pub const MODEL_ID: &str = "wavtokenizer-large-speech-75token";
/// Output sample rate.
pub const SAMPLE_RATE: u32 = 24_000;
/// Samples represented by one token.
pub const HOP_LENGTH: usize = 320;
/// Public checkpoint vocabulary size.
pub const CODEBOOK_SIZE: usize = 4096;
/// Quantized feature width.
pub const FEATURE_DIM: usize = 512;
/// WavTokenizer Vocos hidden width.
pub const HIDDEN_DIM: usize = 768;
/// Number of released decoder condition embeddings.
pub const BANDWIDTH_CONDITIONS: usize = 4;
const INTERMEDIATE_DIM: usize = 2304;
const CONVNEXT_LAYERS: usize = 12;
const N_FFT: usize = 1280;
const POS_GROUPS: usize = 32;
const PUBLIC_TENSOR_COUNT: usize = 1091;

/// Complete learned-op set for public token-to-waveform execution.
pub const WAVTOKENIZER_DECODE_HOT_OPS: &[HotOp] = &[
    HotOp::WavTokenizerVq,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::GroupNorm,
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Silu,
];

#[derive(Debug, Clone)]
struct AffineNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Debug, Clone)]
struct Conv1dWeights {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Debug, Clone)]
struct PosResBlock {
    norm1: AffineNorm,
    conv1: Conv1dWeights,
    norm2: AffineNorm,
    conv2: Conv1dWeights,
}

#[derive(Debug, Clone)]
struct PosAttention {
    norm: AffineNorm,
    q: Conv1dWeights,
    k: Conv1dWeights,
    v: Conv1dWeights,
    proj_out: Conv1dWeights,
}

#[derive(Debug, Clone)]
struct PosNetWeights {
    before_attention: [PosResBlock; 2],
    attention: PosAttention,
    after_attention: [PosResBlock; 2],
    final_norm: AffineNorm,
}

/// Strict real-weight WavTokenizer token-to-PCM model.
#[derive(Debug, Clone)]
pub struct WavTokenizer {
    vocos: VocosWeights,
    pos_net: PosNetWeights,
    codebook: CodebookTable,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl WavTokenizer {
    /// Loads the one released large-speech 75 token/s topology.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        require_string(file, chunks::KEY_MODEL_NAME, MODEL_ID)?;
        require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, MODEL_ID)?;
        require_string(
            file,
            "vokra.provenance.upstream_hf",
            "novateur/WavTokenizer-large-speech-75token",
        )?;
        if file.tensors().len() != PUBLIC_TENSOR_COUNT {
            return Err(VokraError::ModelLoad(format!(
                "wavtokenizer: public checkpoint has {} tensors, expected {PUBLIC_TENSOR_COUNT}; only the audited large-speech-75token release is supported",
                file.tensors().len()
            )));
        }

        validate_encoder_manifest(file)?;
        validate_quantizer_manifest(file)?;
        let mut decoder_names = BTreeSet::new();
        let vocos = load_vocos_weights(file, &mut decoder_names)?;
        let pos_net = load_pos_net(file, &mut decoder_names)?;
        validate_exact_prefix_manifest(file, &["backbone.", "head."], &decoder_names, "decoder")?;

        let codebook_data = load_tensor(
            file,
            "feature_extractor.encodec.quantizer.vq.layers.0._codebook.embed",
            &[CODEBOOK_SIZE, FEATURE_DIM],
            &mut BTreeSet::new(),
        )?;
        let codebook = CodebookTable::new(CODEBOOK_SIZE, FEATURE_DIM, codebook_data)
            .map_err(|error| VokraError::ModelLoad(format!("wavtokenizer codebook: {error}")))?;
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|value| value.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            vocos,
            pos_net,
            codebook,
            weight_license,
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and binds a public GGUF.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Selects one backend for the complete decode graph.
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

    /// Stamped artifact license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Output sample rate.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Samples represented by one code.
    #[must_use]
    pub const fn hop_length(&self) -> usize {
        HOP_LENGTH
    }

    /// Number of public codebook entries.
    #[must_use]
    pub const fn codebook_size(&self) -> usize {
        CODEBOOK_SIZE
    }

    /// Decodes with upstream's documented inference condition id `0`.
    pub fn decode_codes(&self, codes: &[u32]) -> Result<Vec<f32>> {
        self.decode_codes_with_condition(codes, 0)
    }

    /// Decodes tokens using one of the four released AdaLayerNorm condition
    /// rows. The public upstream examples use `0`; exposing the id avoids
    /// silently discarding a caller's conditioning contract.
    pub fn decode_codes_with_condition(
        &self,
        codes: &[u32],
        condition_id: usize,
    ) -> Result<Vec<f32>> {
        if codes.is_empty() {
            return Err(VokraError::InvalidArgument(
                "wavtokenizer: codes must not be empty".to_owned(),
            ));
        }
        if condition_id >= BANDWIDTH_CONDITIONS {
            return Err(VokraError::InvalidArgument(format!(
                "wavtokenizer: condition_id {condition_id} is outside 0..{BANDWIDTH_CONDITIONS}"
            )));
        }
        let frames = codes.len();
        let compute = Compute::for_backend(self.backend, WAVTOKENIZER_DECODE_HOT_OPS)?;
        let attrs = WavTokenizerVqAttrs::wavtokenizer_24k_4096();
        let frame_major = compute.wavtokenizer_vq_f32(codes, frames, &self.codebook, &attrs)?;
        let features = transpose_frame_to_channel(&frame_major, frames, FEATURE_DIM);

        let mut embedded = vec![0.0f32; HIDDEN_DIM * frames];
        compute.conv1d_f32(
            &features,
            FEATURE_DIM,
            frames,
            &self.vocos.embed_weight,
            HIDDEN_DIM,
            7,
            Some(&self.vocos.embed_bias),
            1,
            3,
            &mut embedded,
        )?;
        positional_forward(&compute, &mut embedded, frames, &self.pos_net)?;

        let ops = ComputeVocosOps { compute: &compute };
        vocos_decode_from_embedded_with_ops(
            embedded,
            frames,
            Some(condition_id),
            &self.vocos,
            &vocos_attrs(),
            &ops,
        )
    }
}

fn vocos_attrs() -> VocosAttrs {
    VocosAttrs {
        input_channels: FEATURE_DIM,
        dim: HIDDEN_DIM,
        intermediate_dim: INTERMEDIATE_DIM,
        num_layers: CONVNEXT_LAYERS,
        num_conditions: BANDWIDTH_CONDITIONS,
        n_fft: N_FFT,
        hop_length: HOP_LENGTH,
        padding: VocosIstftPadding::Same,
    }
}

fn positional_forward(
    compute: &Compute,
    values: &mut Vec<f32>,
    frames: usize,
    weights: &PosNetWeights,
) -> Result<()> {
    for block in &weights.before_attention {
        *values = residual_forward(compute, values, frames, block)?;
    }
    *values = attention_forward(compute, values, frames, &weights.attention)?;
    for block in &weights.after_attention {
        *values = residual_forward(compute, values, frames, block)?;
    }
    *values = group_norm(compute, values, frames, &weights.final_norm)?;
    Ok(())
}

fn residual_forward(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    weights: &PosResBlock,
) -> Result<Vec<f32>> {
    let mut hidden = group_norm(compute, input, frames, &weights.norm1)?;
    hidden = silu(compute, &hidden)?;
    hidden = conv_same(compute, &hidden, frames, &weights.conv1, 3)?;
    hidden = group_norm(compute, &hidden, frames, &weights.norm2)?;
    hidden = silu(compute, &hidden)?;
    hidden = conv_same(compute, &hidden, frames, &weights.conv2, 3)?;
    for (output, &residual) in hidden.iter_mut().zip(input) {
        *output += residual;
    }
    Ok(hidden)
}

fn attention_forward(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    weights: &PosAttention,
) -> Result<Vec<f32>> {
    let normalized = group_norm(compute, input, frames, &weights.norm)?;
    let q = conv_same(compute, &normalized, frames, &weights.q, 1)?;
    let k = conv_same(compute, &normalized, frames, &weights.k, 1)?;
    let v = conv_same(compute, &normalized, frames, &weights.v, 1)?;
    let q_frame_major = transpose_channel_to_frame(&q, HIDDEN_DIM, frames);
    let mut logits = vec![0.0f32; frames * frames];
    compute.gemm_f32(
        frames,
        frames,
        HIDDEN_DIM,
        &q_frame_major,
        &k,
        None,
        &mut logits,
    )?;
    let scale = (HIDDEN_DIM as f32).sqrt().recip();
    for value in &mut logits {
        *value *= scale;
    }
    let mut probabilities = vec![0.0f32; logits.len()];
    compute.softmax_f32(&logits, &mut probabilities, frames, frames)?;
    let probabilities_t = transpose_square(&probabilities, frames);
    let mut attended = vec![0.0f32; HIDDEN_DIM * frames];
    compute.gemm_f32(
        HIDDEN_DIM,
        frames,
        frames,
        &v,
        &probabilities_t,
        None,
        &mut attended,
    )?;
    let projected = conv_same(compute, &attended, frames, &weights.proj_out, 1)?;
    Ok(input
        .iter()
        .zip(projected)
        .map(|(&residual, value)| residual + value)
        .collect())
}

fn group_norm(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    norm: &AffineNorm,
) -> Result<Vec<f32>> {
    let channels_per_group = HIDDEN_DIM / POS_GROUPS;
    let values_per_group = channels_per_group * frames;
    let mut output = vec![0.0f32; input.len()];
    for group in 0..POS_GROUPS {
        let value_start = group * values_per_group;
        let channel_start = group * channels_per_group;
        compute.group_norm_f32(
            &input[value_start..value_start + values_per_group],
            &mut output[value_start..value_start + values_per_group],
            channels_per_group,
            frames,
            &norm.weight[channel_start..channel_start + channels_per_group],
            &norm.bias[channel_start..channel_start + channels_per_group],
            1e-6,
        )?;
    }
    Ok(output)
}

fn silu(compute: &Compute, input: &[f32]) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; input.len()];
    compute.silu_f32(input, &mut output)?;
    Ok(output)
}

fn conv_same(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    weights: &Conv1dWeights,
    kernel: usize,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; HIDDEN_DIM * frames];
    compute.conv1d_f32(
        input,
        HIDDEN_DIM,
        frames,
        &weights.weight,
        HIDDEN_DIM,
        kernel,
        Some(&weights.bias),
        1,
        kernel / 2,
        &mut output,
    )?;
    Ok(output)
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
        let frame_major = transpose_channel_to_frame(values, dim, frames);
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
        values.copy_from_slice(&transpose_frame_to_channel(&normalized, frames, dim));
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

fn load_vocos_weights(file: &GgufFile, names: &mut BTreeSet<String>) -> Result<VocosWeights> {
    let attrs = vocos_attrs();
    let embed_weight = load_tensor(
        file,
        "backbone.embed.weight",
        &[HIDDEN_DIM, FEATURE_DIM, 7],
        names,
    )?;
    let embed_bias = load_tensor(file, "backbone.embed.bias", &[HIDDEN_DIM], names)?;
    let norm = load_vocos_norm(file, "backbone.norm", names)?;
    let mut blocks = Vec::with_capacity(CONVNEXT_LAYERS);
    for index in 0..CONVNEXT_LAYERS {
        let prefix = format!("backbone.convnext.{index}");
        blocks.push(VocosBlockWeights {
            depthwise_weight: load_tensor(
                file,
                &format!("{prefix}.dwconv.weight"),
                &[HIDDEN_DIM, 1, 7],
                names,
            )?,
            depthwise_bias: load_tensor(
                file,
                &format!("{prefix}.dwconv.bias"),
                &[HIDDEN_DIM],
                names,
            )?,
            norm: load_vocos_norm(file, &format!("{prefix}.norm"), names)?,
            pointwise1_weight: load_tensor(
                file,
                &format!("{prefix}.pwconv1.weight"),
                &[INTERMEDIATE_DIM, HIDDEN_DIM],
                names,
            )?,
            pointwise1_bias: load_tensor(
                file,
                &format!("{prefix}.pwconv1.bias"),
                &[INTERMEDIATE_DIM],
                names,
            )?,
            pointwise2_weight: load_tensor(
                file,
                &format!("{prefix}.pwconv2.weight"),
                &[HIDDEN_DIM, INTERMEDIATE_DIM],
                names,
            )?,
            pointwise2_bias: load_tensor(
                file,
                &format!("{prefix}.pwconv2.bias"),
                &[HIDDEN_DIM],
                names,
            )?,
            gamma: load_tensor(file, &format!("{prefix}.gamma"), &[HIDDEN_DIM], names)?,
        });
    }
    let final_norm_weight = load_tensor(
        file,
        "backbone.final_layer_norm.weight",
        &[HIDDEN_DIM],
        names,
    )?;
    let final_norm_bias =
        load_tensor(file, "backbone.final_layer_norm.bias", &[HIDDEN_DIM], names)?;
    let head_weight = load_tensor(file, "head.out.weight", &[N_FFT + 2, HIDDEN_DIM], names)?;
    let head_bias = load_tensor(file, "head.out.bias", &[N_FFT + 2], names)?;
    let _window = load_tensor(file, "head.istft.window", &[N_FFT], names)?;
    let weights = VocosWeights {
        embed_weight,
        embed_bias,
        norm,
        blocks,
        final_norm_weight,
        final_norm_bias,
        head_weight,
        head_bias,
    };
    weights.validate(&attrs).map_err(|error| {
        VokraError::ModelLoad(format!("wavtokenizer Vocos weight validation: {error}"))
    })?;
    Ok(weights)
}

fn load_vocos_norm(
    file: &GgufFile,
    prefix: &str,
    names: &mut BTreeSet<String>,
) -> Result<VocosNormWeights> {
    Ok(VocosNormWeights {
        scale: load_tensor(
            file,
            &format!("{prefix}.scale.weight"),
            &[BANDWIDTH_CONDITIONS, HIDDEN_DIM],
            names,
        )?,
        shift: load_tensor(
            file,
            &format!("{prefix}.shift.weight"),
            &[BANDWIDTH_CONDITIONS, HIDDEN_DIM],
            names,
        )?,
    })
}

fn load_pos_net(file: &GgufFile, names: &mut BTreeSet<String>) -> Result<PosNetWeights> {
    Ok(PosNetWeights {
        before_attention: [
            load_res_block(file, 0, names)?,
            load_res_block(file, 1, names)?,
        ],
        attention: load_attention(file, names)?,
        after_attention: [
            load_res_block(file, 3, names)?,
            load_res_block(file, 4, names)?,
        ],
        final_norm: load_affine_norm(file, "backbone.pos_net.5", names)?,
    })
}

fn load_res_block(
    file: &GgufFile,
    index: usize,
    names: &mut BTreeSet<String>,
) -> Result<PosResBlock> {
    let prefix = format!("backbone.pos_net.{index}");
    Ok(PosResBlock {
        norm1: load_affine_norm(file, &format!("{prefix}.norm1"), names)?,
        conv1: load_conv(file, &format!("{prefix}.conv1"), 3, names)?,
        norm2: load_affine_norm(file, &format!("{prefix}.norm2"), names)?,
        conv2: load_conv(file, &format!("{prefix}.conv2"), 3, names)?,
    })
}

fn load_attention(file: &GgufFile, names: &mut BTreeSet<String>) -> Result<PosAttention> {
    let prefix = "backbone.pos_net.2";
    Ok(PosAttention {
        norm: load_affine_norm(file, &format!("{prefix}.norm"), names)?,
        q: load_conv(file, &format!("{prefix}.q"), 1, names)?,
        k: load_conv(file, &format!("{prefix}.k"), 1, names)?,
        v: load_conv(file, &format!("{prefix}.v"), 1, names)?,
        proj_out: load_conv(file, &format!("{prefix}.proj_out"), 1, names)?,
    })
}

fn load_affine_norm(
    file: &GgufFile,
    prefix: &str,
    names: &mut BTreeSet<String>,
) -> Result<AffineNorm> {
    Ok(AffineNorm {
        weight: load_tensor(file, &format!("{prefix}.weight"), &[HIDDEN_DIM], names)?,
        bias: load_tensor(file, &format!("{prefix}.bias"), &[HIDDEN_DIM], names)?,
    })
}

fn load_conv(
    file: &GgufFile,
    prefix: &str,
    kernel: usize,
    names: &mut BTreeSet<String>,
) -> Result<Conv1dWeights> {
    Ok(Conv1dWeights {
        weight: load_tensor(
            file,
            &format!("{prefix}.weight"),
            &[HIDDEN_DIM, HIDDEN_DIM, kernel],
            names,
        )?,
        bias: load_tensor(file, &format!("{prefix}.bias"), &[HIDDEN_DIM], names)?,
    })
}

fn validate_encoder_manifest(file: &GgufFile) -> Result<()> {
    let prefix = "feature_extractor.encodec.encoder.";
    let mut expected = BTreeSet::new();
    require_weight_norm_conv(
        file,
        "feature_extractor.encodec.encoder.model.0.conv.conv",
        1,
        32,
        7,
        &mut expected,
    )?;
    let channels = [32usize, 64, 128, 256];
    let model_indices = [1usize, 4, 7, 10];
    for (&channel, &index) in channels.iter().zip(&model_indices) {
        let block = format!("feature_extractor.encodec.encoder.model.{index}");
        require_weight_norm_conv(
            file,
            &format!("{block}.block.1.conv.conv"),
            channel,
            channel / 2,
            3,
            &mut expected,
        )?;
        require_weight_norm_conv(
            file,
            &format!("{block}.block.3.conv.conv"),
            channel / 2,
            channel,
            1,
            &mut expected,
        )?;
        require_weight_norm_conv(
            file,
            &format!("{block}.shortcut.conv.conv"),
            channel,
            channel,
            1,
            &mut expected,
        )?;
    }
    for (index, input, output, kernel) in [
        (3usize, 32usize, 64usize, 4usize),
        (6, 64, 128, 8),
        (9, 128, 256, 10),
        (12, 256, 512, 16),
    ] {
        require_weight_norm_conv(
            file,
            &format!("feature_extractor.encodec.encoder.model.{index}.conv.conv"),
            input,
            output,
            kernel,
            &mut expected,
        )?;
    }
    for layer in 0..2 {
        for kind in ["weight_ih", "weight_hh"] {
            require_shape(
                file,
                &format!("feature_extractor.encodec.encoder.model.13.lstm.{kind}_l{layer}"),
                &[2048, 512],
                &mut expected,
            )?;
        }
        for kind in ["bias_ih", "bias_hh"] {
            require_shape(
                file,
                &format!("feature_extractor.encodec.encoder.model.13.lstm.{kind}_l{layer}"),
                &[2048],
                &mut expected,
            )?;
        }
    }
    require_weight_norm_conv(
        file,
        "feature_extractor.encodec.encoder.model.15.conv.conv",
        512,
        512,
        7,
        &mut expected,
    )?;
    validate_exact_prefix_manifest(file, &[prefix], &expected, "encoder")
}

fn validate_quantizer_manifest(file: &GgufFile) -> Result<()> {
    let prefix = "feature_extractor.encodec.quantizer.";
    let base = "feature_extractor.encodec.quantizer.vq.layers.0._codebook";
    let mut expected = BTreeSet::new();
    require_shape(
        file,
        &format!("{base}.cluster_size"),
        &[CODEBOOK_SIZE],
        &mut expected,
    )?;
    require_shape(
        file,
        &format!("{base}.embed"),
        &[CODEBOOK_SIZE, FEATURE_DIM],
        &mut expected,
    )?;
    require_shape(
        file,
        &format!("{base}.embed_avg"),
        &[CODEBOOK_SIZE, FEATURE_DIM],
        &mut expected,
    )?;
    require_shape(file, &format!("{base}.inited"), &[1], &mut expected)?;
    validate_exact_prefix_manifest(file, &[prefix], &expected, "quantizer")
}

fn require_weight_norm_conv(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    expected: &mut BTreeSet<String>,
) -> Result<()> {
    require_shape(file, &format!("{prefix}.bias"), &[output], expected)?;
    require_shape(
        file,
        &format!("{prefix}.weight_g"),
        &[output, 1, 1],
        expected,
    )?;
    require_shape(
        file,
        &format!("{prefix}.weight_v"),
        &[output, input, kernel],
        expected,
    )
}

fn validate_exact_prefix_manifest(
    file: &GgufFile,
    prefixes: &[&str],
    expected: &BTreeSet<String>,
    label: &str,
) -> Result<()> {
    let actual: BTreeSet<String> = file
        .tensors()
        .iter()
        .filter(|tensor| {
            prefixes
                .iter()
                .any(|prefix| tensor.name.starts_with(prefix))
        })
        .map(|tensor| tensor.name.clone())
        .collect();
    if actual != *expected {
        let missing: Vec<_> = expected.difference(&actual).take(4).collect();
        let extra: Vec<_> = actual.difference(expected).take(4).collect();
        return Err(VokraError::ModelLoad(format!(
            "wavtokenizer {label}: tensor manifest mismatch (expected {}, found {}); missing={missing:?}, extra={extra:?}",
            expected.len(),
            actual.len()
        )));
    }
    Ok(())
}

fn require_shape(
    file: &GgufFile,
    name: &str,
    expected_shape: &[usize],
    expected: &mut BTreeSet<String>,
) -> Result<()> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("wavtokenizer: required tensor `{name}` is missing"))
    })?;
    let actual_shape: Vec<usize> = info
        .dimensions
        .iter()
        .map(|&value| value as usize)
        .collect();
    if actual_shape != expected_shape {
        return Err(VokraError::ModelLoad(format!(
            "wavtokenizer: tensor `{name}` shape {actual_shape:?}, expected {expected_shape:?}"
        )));
    }
    expected.insert(name.to_owned());
    Ok(())
}

fn load_tensor(
    file: &GgufFile,
    name: &str,
    expected_shape: &[usize],
    expected: &mut BTreeSet<String>,
) -> Result<Vec<f32>> {
    require_shape(file, name, expected_shape, expected)?;
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!(
            "wavtokenizer: tensor `{name}` decode failed: {error}"
        ))
    })
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(|value| value.as_str());
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "wavtokenizer: metadata `{key}` = {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn transpose_frame_to_channel(input: &[f32], frames: usize, channels: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for frame in 0..frames {
        for channel in 0..channels {
            output[channel * frames + frame] = input[frame * channels + channel];
        }
    }
    output
}

fn transpose_channel_to_frame(input: &[f32], channels: usize, frames: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for channel in 0..channels {
        for frame in 0..frames {
            output[frame * channels + channel] = input[channel * frames + frame];
        }
    }
    output
}

fn transpose_square(input: &[f32], size: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for row in 0..size {
        for column in 0..size {
            output[column * size + row] = input[row * size + column];
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_helpers_preserve_axes() {
        let frame_major = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let channel_major = transpose_frame_to_channel(&frame_major, 2, 3);
        assert_eq!(channel_major, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(
            transpose_channel_to_frame(&channel_major, 3, 2),
            frame_major
        );
        assert_eq!(
            transpose_square(&[1.0, 2.0, 3.0, 4.0], 2),
            vec![1.0, 3.0, 2.0, 4.0]
        );
    }

    #[test]
    fn decoder_hot_ops_are_cpu_and_metal_complete() {
        Compute::for_backend(BackendKind::Cpu, WAVTOKENIZER_DECODE_HOT_OPS)
            .expect("CPU covers the full WavTokenizer decode");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, WAVTOKENIZER_DECODE_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("WavTokenizer decode has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    fn public_topology_constants_are_consistent() {
        assert_eq!(SAMPLE_RATE as usize / HOP_LENGTH, 75);
        assert_eq!(HIDDEN_DIM % POS_GROUPS, 0);
        assert_eq!(vocos_attrs().num_layers, CONVNEXT_LAYERS);
        assert_eq!(vocos_attrs().num_conditions, BANDWIDTH_CONDITIONS);
    }
}
