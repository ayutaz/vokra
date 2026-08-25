//! Native NeuCodec token-to-waveform decoder.
//!
//! The public `vokra/neucodec` and `vokra/distill-neucodec` checkpoints use
//! the same released decoder topology. The encoder halves differ, but decode
//! is shared: 65,536-way FSQ -> 2048-to-1024 projection -> convolutional
//! residual pre-net -> twelve non-causal Transformer blocks -> residual
//! post-net -> magnitude/phase iSTFT at 24 kHz.
//!
//! Every learned operation routes through [`Compute`]. CPU is the scalar
//! reference and Apple Metal uses the already-wired FSQ, convolution, GEMM,
//! softmax and normalization kernels. Unsupported backends fail at the
//! whole-model coverage gate; there is no silent CPU fallback.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::ir::graph::IstftAttrs;
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::{FsqOutProj, Spectrogram, Xcodec2FsqAttrs, istft};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec, load_tensor};

/// Converter/runtime architecture tag shared by both official variants.
pub const ARCH: &str = "neucodec";
/// Decoder output sample rate.
pub const SAMPLE_RATE: u32 = 24_000;
/// Samples represented by one 50 Hz code.
pub const HOP_LENGTH: usize = 480;
/// Product of the released `[4; 8]` FSQ levels.
pub const CODEBOOK_SIZE: usize = 65_536;
/// Width emitted by the FSQ output projection.
pub const QUANTIZED_DIM: usize = 2_048;
/// Decoder Transformer and convolution width.
pub const HIDDEN_DIM: usize = 1_024;

const FSQ_DIM: usize = 8;
const TRANSFORMER_LAYERS: usize = 12;
const ATTENTION_HEADS: usize = 16;
const HEAD_DIM: usize = HIDDEN_DIM / ATTENTION_HEADS;
const INTERMEDIATE_DIM: usize = HIDDEN_DIM * 4;
const RESNET_GROUPS: usize = 32;
const N_FFT: usize = HOP_LENGTH * 4;
const HEAD_OUTPUT_DIM: usize = N_FFT + 2;
const NORM_EPS: f32 = 1.0e-6;
const ROPE_BASE: f32 = 10_000.0;

const DISTILL_SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: "distill-neucodec",
    arch: ARCH,
    model_name: "distill-neucodec",
    model_name_alias: None,
    tensor_count: 294,
    manifest_sha256: [
        0x8b, 0xf4, 0xf1, 0x71, 0x55, 0x9b, 0x9d, 0xa0, 0xd1, 0x53, 0x18, 0x67, 0xa7, 0xf2, 0xbf,
        0xec, 0x52, 0x65, 0xcc, 0x59, 0x32, 0xb0, 0xdf, 0x89, 0x5a, 0x51, 0x91, 0x34, 0x38, 0x74,
        0x4f, 0x1b,
    ],
};

// The base artifact predates the pass-through converter used for Distill. It
// contains the full encoders and a normalized decoder namespace, so its
// complete public manifest is pinned separately.
const BASE_SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: "neucodec",
    arch: ARCH,
    model_name: "neucodec",
    model_name_alias: None,
    tensor_count: 811,
    manifest_sha256: [
        0x1b, 0x76, 0xdc, 0x8f, 0x93, 0xc5, 0xc6, 0x8f, 0x01, 0x32, 0x9f, 0x9f, 0x05, 0xb6, 0xf3,
        0x42, 0x92, 0xb4, 0x1b, 0xd3, 0x9b, 0x4c, 0x46, 0xe0, 0x82, 0x29, 0x32, 0x7d, 0xaa, 0x01,
        0x02, 0xe0,
    ],
};

/// Complete learned-op set for official token-to-waveform execution.
pub const NEUCODEC_DECODE_HOT_OPS: &[HotOp] = &[
    HotOp::Xcodec2Fsq,
    HotOp::Conv1d,
    HotOp::GroupNorm,
    HotOp::RmsNorm,
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::Silu,
    HotOp::LayerNorm,
];

/// Official checkpoint variant. Both variants share the exact decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuCodecVariant {
    /// `vokra/neucodec`, using the full acoustic and semantic encoders.
    Base,
    /// `vokra/distill-neucodec`, using the smaller distilled encoders.
    Distill,
}

impl NeuCodecVariant {
    const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Base => "neuphonic/neucodec",
            Self::Distill => "neuphonic/distill-neucodec",
        }
    }

    const fn spec(self) -> StrictCheckpointSpec {
        match self {
            Self::Base => BASE_SPEC,
            Self::Distill => DISTILL_SPEC,
        }
    }
}

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
struct BiaslessLinear {
    weight: Vec<f32>,
    input_dim: usize,
    output_dim: usize,
}

#[derive(Debug, Clone)]
struct ResnetBlock {
    norm1: AffineNorm,
    conv1: Conv1dWeights,
    norm2: AffineNorm,
    conv2: Conv1dWeights,
}

#[derive(Debug, Clone)]
struct TransformerBlock {
    att_norm: Vec<f32>,
    c_attn: BiaslessLinear,
    c_proj: BiaslessLinear,
    ffn_norm: Vec<f32>,
    fc1: BiaslessLinear,
    fc2: BiaslessLinear,
}

#[derive(Debug, Clone)]
struct DecoderWeights {
    fsq_out_proj: FsqOutProj,
    fc_post_a: Conv1dWeights,
    embed: Conv1dWeights,
    prior_net: [ResnetBlock; 2],
    transformers: Vec<TransformerBlock>,
    post_net: [ResnetBlock; 2],
    final_norm: AffineNorm,
    head: Conv1dWeights,
}

/// Strict real-weight NeuCodec token-to-PCM model.
#[derive(Debug, Clone)]
pub struct NeuCodec {
    weights: DecoderWeights,
    variant: NeuCodecVariant,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl NeuCodec {
    /// Binds either audited public Vokra NeuCodec checkpoint.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let variant = match file
            .get("vokra.neucodec.variant")
            .and_then(GgufMetadataValue::as_str)
        {
            Some("base") => NeuCodecVariant::Base,
            Some("distill") => NeuCodecVariant::Distill,
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "neucodec: unsupported `vokra.neucodec.variant`={other:?}; expected \"base\" or \"distill\""
                )));
            }
            // The first public base GGUF predates the additive variant key.
            // Its full manifest and model identity below still make this
            // legacy recognition fail closed.
            None if required_string(file, chunks::KEY_MODEL_NAME)? == "neucodec" => {
                NeuCodecVariant::Base
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "neucodec: missing/non-string `vokra.neucodec.variant` for non-base model"
                        .to_owned(),
                ));
            }
        };
        let checkpoint = StrictCheckpoint::bind(file, variant.spec())?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_MODEL_ID,
            checkpoint.model_name(),
        )?;
        require_string(file, "vokra.provenance.upstream_hf", variant.upstream_hf())?;
        let weights = load_decoder(file, variant)?;
        Ok(Self {
            weights,
            variant,
            weight_license: checkpoint.weight_license(),
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and binds an official GGUF. The CLI's session path uses mmap;
    /// this convenience method retains the core buffered-reader semantics.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Selects one backend for the complete decoder graph.
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

    /// Bound official variant.
    #[must_use]
    pub const fn variant(&self) -> NeuCodecVariant {
        self.variant
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

    /// Decodes one batch-free `[frames]` FSQ code sequence to 24 kHz PCM.
    pub fn decode_codes(&self, codes: &[u32]) -> Result<Vec<f32>> {
        if codes.is_empty() {
            return Err(VokraError::InvalidArgument(
                "neucodec: codes must not be empty".to_owned(),
            ));
        }
        let frames = codes.len();
        let _ = frames.checked_mul(HIDDEN_DIM).ok_or_else(|| {
            VokraError::InvalidArgument("neucodec: frame activation size overflow".to_owned())
        })?;
        let expected_samples = frames.checked_mul(HOP_LENGTH).ok_or_else(|| {
            VokraError::InvalidArgument("neucodec: output sample count overflow".to_owned())
        })?;
        let compute = Compute::for_backend(self.backend, NEUCODEC_DECODE_HOT_OPS)?;
        let attrs = Xcodec2FsqAttrs {
            levels: vec![4; FSQ_DIM],
            d_model: QUANTIZED_DIM,
        };
        let quantized =
            compute.xcodec2_fsq_f32(codes, frames, Some(&self.weights.fsq_out_proj), &attrs)?;
        let quantized = transpose_frame_to_channel(&quantized, frames, QUANTIZED_DIM);
        let mut hidden = conv1d_same(
            &compute,
            &quantized,
            QUANTIZED_DIM,
            frames,
            HIDDEN_DIM,
            &self.weights.fc_post_a,
            1,
        )?;
        hidden = conv1d_same(
            &compute,
            &hidden,
            HIDDEN_DIM,
            frames,
            HIDDEN_DIM,
            &self.weights.embed,
            7,
        )?;
        for block in &self.weights.prior_net {
            hidden = resnet_forward(&compute, &hidden, frames, block)?;
        }
        for block in &self.weights.transformers {
            hidden = transformer_forward(&compute, &hidden, frames, block)?;
        }
        for block in &self.weights.post_net {
            hidden = resnet_forward(&compute, &hidden, frames, block)?;
        }
        let frame_major = transpose_channel_to_frame(&hidden, HIDDEN_DIM, frames);
        let mut normalized = vec![0.0f32; frame_major.len()];
        compute.layer_norm_f32(
            &frame_major,
            &mut normalized,
            frames,
            HIDDEN_DIM,
            &self.weights.final_norm.weight,
            &self.weights.final_norm.bias,
            NORM_EPS,
        )?;
        let normalized = transpose_frame_to_channel(&normalized, frames, HIDDEN_DIM);
        let projected = conv1d_same(
            &compute,
            &normalized,
            HIDDEN_DIM,
            frames,
            HEAD_OUTPUT_DIM,
            &self.weights.head,
            1,
        )?;
        let pcm = istft_head(&projected, frames)?;
        if pcm.len() != expected_samples {
            return Err(VokraError::InvalidArgument(format!(
                "neucodec: decoder emitted {} samples, expected {expected_samples}",
                pcm.len()
            )));
        }
        Ok(pcm)
    }
}

fn resnet_forward(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    weights: &ResnetBlock,
) -> Result<Vec<f32>> {
    let mut hidden = group_norm(compute, input, frames, &weights.norm1)?;
    hidden = silu(compute, &hidden)?;
    hidden = conv1d_same(
        compute,
        &hidden,
        HIDDEN_DIM,
        frames,
        HIDDEN_DIM,
        &weights.conv1,
        3,
    )?;
    hidden = group_norm(compute, &hidden, frames, &weights.norm2)?;
    hidden = silu(compute, &hidden)?;
    hidden = conv1d_same(
        compute,
        &hidden,
        HIDDEN_DIM,
        frames,
        HIDDEN_DIM,
        &weights.conv2,
        3,
    )?;
    add_residual(&mut hidden, input);
    Ok(hidden)
}

fn transformer_forward(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    weights: &TransformerBlock,
) -> Result<Vec<f32>> {
    let frame_major = transpose_channel_to_frame(input, HIDDEN_DIM, frames);
    let mut normalized = vec![0.0f32; frame_major.len()];
    compute.rms_norm_f32(
        &frame_major,
        &mut normalized,
        frames,
        HIDDEN_DIM,
        &weights.att_norm,
        NORM_EPS,
    )?;
    let normalized = transpose_frame_to_channel(&normalized, frames, HIDDEN_DIM);
    let qkv = biasless_pointwise(compute, &normalized, frames, &weights.c_attn)?;
    let mut attended = vec![0.0f32; HIDDEN_DIM * frames];
    let scale = (HEAD_DIM as f32).sqrt().recip();
    for head in 0..ATTENTION_HEADS {
        let mut q = vec![0.0f32; frames * HEAD_DIM];
        let mut k = vec![0.0f32; frames * HEAD_DIM];
        let mut v = vec![0.0f32; frames * HEAD_DIM];
        for frame in 0..frames {
            for dim in 0..HEAD_DIM {
                let channel = head * HEAD_DIM + dim;
                q[frame * HEAD_DIM + dim] = qkv[channel * frames + frame];
                k[frame * HEAD_DIM + dim] = qkv[(HIDDEN_DIM + channel) * frames + frame];
                v[frame * HEAD_DIM + dim] = qkv[(2 * HIDDEN_DIM + channel) * frames + frame];
            }
        }
        apply_official_head_axis_rope(&mut q, frames, head);
        apply_official_head_axis_rope(&mut k, frames, head);
        let k_t = transpose_frame_to_channel(&k, frames, HEAD_DIM);
        let mut logits = vec![0.0f32; frames * frames];
        compute.gemm_f32(frames, frames, HEAD_DIM, &q, &k_t, None, &mut logits)?;
        for value in &mut logits {
            *value *= scale;
        }
        let mut probabilities = vec![0.0f32; logits.len()];
        compute.softmax_f32(&logits, &mut probabilities, frames, frames)?;
        let mut context = vec![0.0f32; frames * HEAD_DIM];
        compute.gemm_f32(
            frames,
            HEAD_DIM,
            frames,
            &probabilities,
            &v,
            None,
            &mut context,
        )?;
        for frame in 0..frames {
            for dim in 0..HEAD_DIM {
                attended[(head * HEAD_DIM + dim) * frames + frame] =
                    context[frame * HEAD_DIM + dim];
            }
        }
    }
    let projected = biasless_pointwise(compute, &attended, frames, &weights.c_proj)?;
    let mut after_attention = input.to_vec();
    add_residual(&mut after_attention, &projected);

    let frame_major = transpose_channel_to_frame(&after_attention, HIDDEN_DIM, frames);
    let mut normalized = vec![0.0f32; frame_major.len()];
    compute.rms_norm_f32(
        &frame_major,
        &mut normalized,
        frames,
        HIDDEN_DIM,
        &weights.ffn_norm,
        NORM_EPS,
    )?;
    let normalized = transpose_frame_to_channel(&normalized, frames, HIDDEN_DIM);
    let mut hidden = biasless_pointwise(compute, &normalized, frames, &weights.fc1)?;
    hidden = silu(compute, &hidden)?;
    let output = biasless_pointwise(compute, &hidden, frames, &weights.fc2)?;
    add_residual(&mut after_attention, &output);
    Ok(after_attention)
}

/// The pinned upstream calls torchtune 0.3.1 RoPE with `[B,H,T,D]` even
/// though that implementation documents `[B,S,H,D]`. Consequently the cache
/// position is the attention-head index and is constant across time. Preserve
/// that released forward exactly; silently applying ordinary time-axis RoPE
/// produces plausible but wrong audio.
fn apply_official_head_axis_rope(values: &mut [f32], frames: usize, head: usize) {
    for pair in 0..HEAD_DIM / 2 {
        let exponent = (2 * pair) as f32 / HEAD_DIM as f32;
        let angle = head as f32 / ROPE_BASE.powf(exponent);
        let cos = angle.cos();
        let sin = angle.sin();
        for frame in 0..frames {
            let offset = frame * HEAD_DIM + 2 * pair;
            let left = values[offset];
            let right = values[offset + 1];
            values[offset] = left * cos - right * sin;
            values[offset + 1] = right * cos + left * sin;
        }
    }
}

fn group_norm(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    weights: &AffineNorm,
) -> Result<Vec<f32>> {
    let channels_per_group = HIDDEN_DIM / RESNET_GROUPS;
    let values_per_group = channels_per_group * frames;
    let mut output = vec![0.0f32; input.len()];
    for group in 0..RESNET_GROUPS {
        let value_start = group * values_per_group;
        let channel_start = group * channels_per_group;
        compute.group_norm_f32(
            &input[value_start..value_start + values_per_group],
            &mut output[value_start..value_start + values_per_group],
            channels_per_group,
            frames,
            &weights.weight[channel_start..channel_start + channels_per_group],
            &weights.bias[channel_start..channel_start + channels_per_group],
            NORM_EPS,
        )?;
    }
    Ok(output)
}

fn silu(compute: &Compute, input: &[f32]) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; input.len()];
    compute.silu_f32(input, &mut output)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn conv1d_same(
    compute: &Compute,
    input: &[f32],
    input_channels: usize,
    frames: usize,
    output_channels: usize,
    weights: &Conv1dWeights,
    kernel: usize,
) -> Result<Vec<f32>> {
    let output_len = output_channels.checked_mul(frames).ok_or_else(|| {
        VokraError::InvalidArgument("neucodec: convolution output size overflow".to_owned())
    })?;
    let mut output = vec![0.0f32; output_len];
    compute.conv1d_f32(
        input,
        input_channels,
        frames,
        &weights.weight,
        output_channels,
        kernel,
        Some(&weights.bias),
        1,
        kernel / 2,
        &mut output,
    )?;
    Ok(output)
}

fn biasless_pointwise(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    weights: &BiaslessLinear,
) -> Result<Vec<f32>> {
    let output_len = weights.output_dim.checked_mul(frames).ok_or_else(|| {
        VokraError::InvalidArgument("neucodec: pointwise output size overflow".to_owned())
    })?;
    let mut output = vec![0.0f32; output_len];
    compute.conv1d_f32(
        input,
        weights.input_dim,
        frames,
        &weights.weight,
        weights.output_dim,
        1,
        None,
        1,
        0,
        &mut output,
    )?;
    Ok(output)
}

fn add_residual(output: &mut [f32], residual: &[f32]) {
    debug_assert_eq!(output.len(), residual.len());
    for (value, &skip) in output.iter_mut().zip(residual) {
        *value += skip;
    }
}

fn istft_head(projected: &[f32], frames: usize) -> Result<Vec<f32>> {
    let bins = N_FFT / 2 + 1;
    let mut re = vec![0.0f32; frames * bins];
    let mut im = vec![0.0f32; frames * bins];
    for frame in 0..frames {
        for bin in 0..bins {
            let magnitude = projected[bin * frames + frame].exp().min(100.0);
            let phase = projected[(bins + bin) * frames + frame];
            re[frame * bins + bin] = magnitude * phase.cos();
            im[frame * bins + bin] = magnitude * phase.sin();
        }
    }
    let spectrogram = Spectrogram {
        frames,
        bins,
        re,
        im,
    };
    let mut attrs = IstftAttrs::new(N_FFT, HOP_LENGTH);
    attrs.center = false;
    let pcm = istft(&spectrogram, &attrs)?;
    let trim = (N_FFT - HOP_LENGTH) / 2;
    if trim * 2 > pcm.len() {
        return Err(VokraError::InvalidArgument(format!(
            "neucodec: same-padding trim {trim} exceeds iSTFT length {}",
            pcm.len()
        )));
    }
    Ok(pcm[trim..pcm.len() - trim].to_vec())
}

fn load_decoder(file: &GgufFile, variant: NeuCodecVariant) -> Result<DecoderWeights> {
    match variant {
        NeuCodecVariant::Base => load_base_decoder(file),
        NeuCodecVariant::Distill => load_distill_decoder(file),
    }
}

fn load_distill_decoder(file: &GgufFile) -> Result<DecoderWeights> {
    let fsq_out_proj = FsqOutProj::new(
        QUANTIZED_DIM,
        FSQ_DIM,
        load(
            file,
            "generator.quantizer.project_out.weight",
            &[QUANTIZED_DIM, FSQ_DIM],
        )?,
        load(
            file,
            "generator.quantizer.project_out.bias",
            &[QUANTIZED_DIM],
        )?,
    )
    .map_err(|error| VokraError::ModelLoad(format!("neucodec FSQ projection: {error}")))?;
    // Encode-only half of ResidualFSQ. The complete-manifest hash already
    // pins the tensors; shape-check them too so a semantically incompatible
    // re-export cannot pass by preserving only names.
    let _project_in_weight = load(
        file,
        "generator.quantizer.project_in.weight",
        &[FSQ_DIM, QUANTIZED_DIM],
    )?;
    let _project_in_bias = load(file, "generator.quantizer.project_in.bias", &[FSQ_DIM])?;
    let fc_post_a = load_pointwise(file, "fc_post_a", HIDDEN_DIM, QUANTIZED_DIM)?;
    let embed = load_conv(file, "generator.backbone.embed", HIDDEN_DIM, HIDDEN_DIM, 7)?;
    let prior_net = [
        load_resnet(file, "generator.backbone.prior_net.0")?,
        load_resnet(file, "generator.backbone.prior_net.1")?,
    ];
    let mut transformers = Vec::with_capacity(TRANSFORMER_LAYERS);
    for layer in 0..TRANSFORMER_LAYERS {
        transformers.push(load_distill_transformer(file, layer)?);
    }
    let post_net = [
        load_resnet(file, "generator.backbone.post_net.0")?,
        load_resnet(file, "generator.backbone.post_net.1")?,
    ];
    let final_norm = load_affine_norm(file, "generator.backbone.final_layer_norm")?;
    let head = load_pointwise(file, "generator.head.out", HEAD_OUTPUT_DIM, HIDDEN_DIM)?;
    let _window = load(file, "generator.head.istft.window", &[N_FFT])?;
    Ok(DecoderWeights {
        fsq_out_proj,
        fc_post_a,
        embed,
        prior_net,
        transformers,
        post_net,
        final_norm,
        head,
    })
}

fn load_base_decoder(file: &GgufFile) -> Result<DecoderWeights> {
    let fsq_out_proj = FsqOutProj::new(
        QUANTIZED_DIM,
        FSQ_DIM,
        load(
            file,
            "quantizer.project_out.weight",
            &[QUANTIZED_DIM, FSQ_DIM],
        )?,
        load(file, "quantizer.project_out.bias", &[QUANTIZED_DIM])?,
    )
    .map_err(|error| VokraError::ModelLoad(format!("neucodec FSQ projection: {error}")))?;
    let _project_in_weight = load(
        file,
        "quantizer.project_in.weight",
        &[FSQ_DIM, QUANTIZED_DIM],
    )?;
    let _project_in_bias = load(file, "quantizer.project_in.bias", &[FSQ_DIM])?;
    let fc_post_a = load_pointwise(file, "acoustic_decoder.fc", HIDDEN_DIM, QUANTIZED_DIM)?;
    let embed = load_conv(file, "acoustic_decoder.embed", HIDDEN_DIM, HIDDEN_DIM, 7)?;
    let prior_net = [
        load_resnet(file, "acoustic_decoder.prior_net.0")?,
        load_resnet(file, "acoustic_decoder.prior_net.1")?,
    ];
    let mut transformers = Vec::with_capacity(TRANSFORMER_LAYERS);
    for layer in 0..TRANSFORMER_LAYERS {
        transformers.push(load_base_transformer(file, layer)?);
    }
    let post_net = [
        load_resnet(file, "acoustic_decoder.post_net.0")?,
        load_resnet(file, "acoustic_decoder.post_net.1")?,
    ];
    let final_norm = load_affine_norm(file, "acoustic_decoder.norm")?;
    let head = load_pointwise(
        file,
        "acoustic_decoder.head.linear",
        HEAD_OUTPUT_DIM,
        HIDDEN_DIM,
    )?;
    Ok(DecoderWeights {
        fsq_out_proj,
        fc_post_a,
        embed,
        prior_net,
        transformers,
        post_net,
        final_norm,
        head,
    })
}

fn load_resnet(file: &GgufFile, prefix: &str) -> Result<ResnetBlock> {
    Ok(ResnetBlock {
        norm1: load_affine_norm(file, &format!("{prefix}.norm1"))?,
        conv1: load_conv(file, &format!("{prefix}.conv1"), HIDDEN_DIM, HIDDEN_DIM, 3)?,
        norm2: load_affine_norm(file, &format!("{prefix}.norm2"))?,
        conv2: load_conv(file, &format!("{prefix}.conv2"), HIDDEN_DIM, HIDDEN_DIM, 3)?,
    })
}

fn load_distill_transformer(file: &GgufFile, layer: usize) -> Result<TransformerBlock> {
    let prefix = format!("generator.backbone.transformers.{layer}");
    Ok(TransformerBlock {
        att_norm: load(file, &format!("{prefix}.att_norm.weight"), &[HIDDEN_DIM])?,
        c_attn: load_biasless_linear(
            file,
            &format!("{prefix}.att.c_attn.weight"),
            HIDDEN_DIM,
            3 * HIDDEN_DIM,
        )?,
        c_proj: load_biasless_linear(
            file,
            &format!("{prefix}.att.c_proj.weight"),
            HIDDEN_DIM,
            HIDDEN_DIM,
        )?,
        ffn_norm: load(file, &format!("{prefix}.ffn_norm.weight"), &[HIDDEN_DIM])?,
        fc1: load_biasless_linear(
            file,
            &format!("{prefix}.mlp.fc1.weight"),
            HIDDEN_DIM,
            INTERMEDIATE_DIM,
        )?,
        fc2: load_biasless_linear(
            file,
            &format!("{prefix}.mlp.fc2.weight"),
            INTERMEDIATE_DIM,
            HIDDEN_DIM,
        )?,
    })
}

fn load_base_transformer(file: &GgufFile, layer: usize) -> Result<TransformerBlock> {
    let prefix = format!("acoustic_decoder.layers.{layer}");
    let attention_prefix = format!("{prefix}.self_attn");
    let mut qkv = Vec::with_capacity(3 * HIDDEN_DIM * HIDDEN_DIM);
    for projection in ["q_proj", "k_proj", "v_proj"] {
        qkv.extend(load(
            file,
            &format!("{attention_prefix}.{projection}.weight"),
            &[HIDDEN_DIM, HIDDEN_DIM],
        )?);
    }
    Ok(TransformerBlock {
        att_norm: load(
            file,
            &format!("{prefix}.input_layernorm.weight"),
            &[HIDDEN_DIM],
        )?,
        c_attn: BiaslessLinear {
            weight: qkv,
            input_dim: HIDDEN_DIM,
            output_dim: 3 * HIDDEN_DIM,
        },
        c_proj: load_biasless_linear(
            file,
            &format!("{attention_prefix}.o_proj.weight"),
            HIDDEN_DIM,
            HIDDEN_DIM,
        )?,
        ffn_norm: load(
            file,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[HIDDEN_DIM],
        )?,
        fc1: load_biasless_linear(
            file,
            &format!("{prefix}.mlp.fc1.weight"),
            HIDDEN_DIM,
            INTERMEDIATE_DIM,
        )?,
        fc2: load_biasless_linear(
            file,
            &format!("{prefix}.mlp.fc2.weight"),
            INTERMEDIATE_DIM,
            HIDDEN_DIM,
        )?,
    })
}

fn load_affine_norm(file: &GgufFile, prefix: &str) -> Result<AffineNorm> {
    Ok(AffineNorm {
        weight: load(file, &format!("{prefix}.weight"), &[HIDDEN_DIM])?,
        bias: load(file, &format!("{prefix}.bias"), &[HIDDEN_DIM])?,
    })
}

fn load_conv(
    file: &GgufFile,
    prefix: &str,
    output_dim: usize,
    input_dim: usize,
    kernel: usize,
) -> Result<Conv1dWeights> {
    Ok(Conv1dWeights {
        weight: load(
            file,
            &format!("{prefix}.weight"),
            &[output_dim, input_dim, kernel],
        )?,
        bias: load(file, &format!("{prefix}.bias"), &[output_dim])?,
    })
}

fn load_pointwise(
    file: &GgufFile,
    prefix: &str,
    output_dim: usize,
    input_dim: usize,
) -> Result<Conv1dWeights> {
    Ok(Conv1dWeights {
        weight: load(file, &format!("{prefix}.weight"), &[output_dim, input_dim])?,
        bias: load(file, &format!("{prefix}.bias"), &[output_dim])?,
    })
}

fn load_biasless_linear(
    file: &GgufFile,
    name: &str,
    input_dim: usize,
    output_dim: usize,
) -> Result<BiaslessLinear> {
    Ok(BiaslessLinear {
        weight: load(file, name, &[output_dim, input_dim])?,
        input_dim,
        output_dim,
    })
}

fn load(file: &GgufFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    load_tensor(file, "neucodec", name, shape)
}

fn required_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("neucodec: missing/non-string `{key}`")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "neucodec: metadata `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn transpose_frame_to_channel(input: &[f32], frames: usize, channels: usize) -> Vec<f32> {
    debug_assert_eq!(input.len(), frames * channels);
    let mut output = vec![0.0f32; input.len()];
    for frame in 0..frames {
        for channel in 0..channels {
            output[channel * frames + frame] = input[frame * channels + channel];
        }
    }
    output
}

fn transpose_channel_to_frame(input: &[f32], channels: usize, frames: usize) -> Vec<f32> {
    debug_assert_eq!(input.len(), frames * channels);
    let mut output = vec![0.0f32; input.len()];
    for channel in 0..channels {
        for frame in 0..frames {
            output[frame * channels + channel] = input[channel * frames + frame];
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_hot_ops_are_cpu_and_metal_complete() {
        Compute::for_backend(BackendKind::Cpu, NEUCODEC_DECODE_HOT_OPS)
            .expect("CPU covers the complete NeuCodec decoder");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, NEUCODEC_DECODE_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("NeuCodec decode has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    fn official_rope_uses_head_axis_not_time_axis() {
        let mut head_zero = [1.0f32, 2.0].repeat(HEAD_DIM / 2 * 2);
        let original = head_zero.clone();
        apply_official_head_axis_rope(&mut head_zero, 2, 0);
        assert_eq!(head_zero, original, "head zero has position angle zero");

        let mut head_one = vec![0.0f32; 2 * HEAD_DIM];
        head_one[0] = 1.0;
        head_one[HEAD_DIM] = 1.0;
        apply_official_head_axis_rope(&mut head_one, 2, 1);
        assert_eq!(head_one[0].to_bits(), head_one[HEAD_DIM].to_bits());
        assert_eq!(head_one[1].to_bits(), head_one[HEAD_DIM + 1].to_bits());
        assert!(head_one[1] > 0.0);
    }

    #[test]
    fn public_decoder_constants_are_exact() {
        assert_eq!(SAMPLE_RATE as usize / HOP_LENGTH, 50);
        assert_eq!(4usize.pow(FSQ_DIM as u32), CODEBOOK_SIZE);
        assert_eq!(ATTENTION_HEADS * HEAD_DIM, HIDDEN_DIM);
        assert_eq!(N_FFT / 2 + 1, HEAD_OUTPUT_DIM / 2);
        assert_eq!(HIDDEN_DIM % RESNET_GROUPS, 0);
    }

    #[test]
    fn transpose_helpers_round_trip() {
        let frame_major = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let channel_major = transpose_frame_to_channel(&frame_major, 2, 3);
        assert_eq!(channel_major, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(
            transpose_channel_to_frame(&channel_major, 3, 2),
            frame_major
        );
    }
}
