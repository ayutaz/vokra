use vokra_core::Result;
use vokra_core::gguf::GgufFile;

use crate::strict_checkpoint::load_tensor;

use super::{ATTENTION_HIDDEN, BLOCKS, ENCODER_CHANNELS, FSMN_CHANNELS, QUERY_KEY_DIM};

const LABEL: &str = "mossformer2-ss-16k";

#[derive(Debug, Clone)]
pub(super) struct Norm {
    pub(super) gamma: Vec<f32>,
    pub(super) beta: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct Linear {
    /// Row-major `[input, output]`, transposed once from PyTorch `[output, input]`.
    pub(super) weight_t: Vec<f32>,
    pub(super) bias: Option<Vec<f32>>,
    pub(super) input: usize,
    pub(super) output: usize,
}

#[derive(Debug, Clone)]
pub(super) struct Conv1d {
    /// PyTorch `[output, input / groups, kernel]` layout.
    pub(super) weight: Vec<f32>,
    pub(super) bias: Option<Vec<f32>>,
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) kernel: usize,
    pub(super) groups: usize,
}

#[derive(Debug, Clone)]
pub(super) enum FfNorm {
    Scale { gain: f32 },
    Layer(Norm),
}

#[derive(Debug, Clone)]
pub(super) struct FfConv {
    pub(super) norm: FfNorm,
    pub(super) linear: Linear,
    pub(super) depthwise: Conv1d,
}

#[derive(Debug, Clone)]
pub(super) struct AttentionLayer {
    pub(super) to_hidden: FfConv,
    pub(super) to_qk: FfConv,
    pub(super) to_out: FfConv,
    pub(super) qk_gamma: Vec<f32>,
    pub(super) qk_beta: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct DenseStage {
    pub(super) conv: Conv1d,
    pub(super) norm: Norm,
    pub(super) slope: Vec<f32>,
    pub(super) dilation: usize,
}

#[derive(Debug, Clone)]
pub(super) struct FsmnCore {
    pub(super) linear: Linear,
    pub(super) project: Linear,
    pub(super) dense: [DenseStage; 2],
}

#[derive(Debug, Clone)]
pub(super) struct GatedFsmnLayer {
    pub(super) input_conv: Conv1d,
    pub(super) input_slope: f32,
    pub(super) norm1: Norm,
    pub(super) to_u: FfConv,
    pub(super) to_v: FfConv,
    pub(super) fsmn: FsmnCore,
    pub(super) norm2: Norm,
    pub(super) output_conv: Conv1d,
}

#[derive(Debug, Clone)]
pub(super) struct MaskWeights {
    pub(super) input_norm: Norm,
    pub(super) input_projection: Conv1d,
    pub(super) position_inv_freq: Vec<f32>,
    pub(super) position_scale: f32,
    pub(super) rotary_freqs: Vec<f32>,
    pub(super) attention: Vec<AttentionLayer>,
    pub(super) fsmn: Vec<GatedFsmnLayer>,
    pub(super) final_norm: Norm,
    pub(super) intra_norm: Norm,
    pub(super) output_slope: f32,
    pub(super) speaker_projection: Conv1d,
    pub(super) output: Conv1d,
    pub(super) output_gate: Conv1d,
    pub(super) mask_projection: Conv1d,
}

#[derive(Debug, Clone)]
pub(super) struct Mossformer2Weights {
    pub(super) encoder: Conv1d,
    pub(super) mask: MaskWeights,
    pub(super) decoder: Vec<f32>,
}

impl Mossformer2Weights {
    pub(super) fn bind(file: &GgufFile) -> Result<Self> {
        let root = "mask_net.mdl.intra_mdl.mossformerM";
        let mut attention = Vec::with_capacity(BLOCKS);
        let mut fsmn = Vec::with_capacity(BLOCKS);
        for layer in 0..BLOCKS {
            attention.push(load_attention(file, &format!("{root}.layers.{layer}"))?);
            fsmn.push(load_fsmn(file, &format!("{root}.fsmn.{layer}"))?);
        }
        Ok(Self {
            encoder: load_conv(file, "enc.conv1d", 1, ENCODER_CHANNELS, 16, 1, false)?,
            decoder: load_tensor(file, LABEL, "dec.weight", &[512, 1, 16])?,
            mask: MaskWeights {
                input_norm: load_norm(file, "mask_net.norm", ENCODER_CHANNELS)?,
                input_projection: load_conv(
                    file,
                    "mask_net.conv1d_encoder",
                    ENCODER_CHANNELS,
                    ENCODER_CHANNELS,
                    1,
                    1,
                    false,
                )?,
                position_inv_freq: load_tensor(file, LABEL, "mask_net.pos_enc.inv_freq", &[256])?,
                position_scale: load_scalar(file, "mask_net.pos_enc.scale")?,
                rotary_freqs: load_tensor(
                    file,
                    LABEL,
                    &format!("{root}.layers.0.rotary_pos_emb.freqs"),
                    &[16],
                )?,
                attention,
                fsmn,
                final_norm: load_norm(file, "mask_net.mdl.intra_mdl.norm", 512)?,
                intra_norm: load_norm(file, "mask_net.mdl.intra_norm", 512)?,
                output_slope: load_scalar(file, "mask_net.prelu.weight")?,
                speaker_projection: load_conv(file, "mask_net.conv1d_out", 512, 1_024, 1, 1, true)?,
                output: load_conv(file, "mask_net.output.0", 512, 512, 1, 1, true)?,
                output_gate: load_conv(file, "mask_net.output_gate.0", 512, 512, 1, 1, true)?,
                mask_projection: load_conv(file, "mask_net.conv1_decoder", 512, 512, 1, 1, false)?,
            },
        })
    }
}

fn load_attention(file: &GgufFile, prefix: &str) -> Result<AttentionLayer> {
    Ok(AttentionLayer {
        to_hidden: load_ff_scale(file, &format!("{prefix}.to_hidden"), 512, ATTENTION_HIDDEN)?,
        to_qk: load_ff_scale(file, &format!("{prefix}.to_qk"), 512, QUERY_KEY_DIM)?,
        to_out: load_ff_scale(file, &format!("{prefix}.to_out"), 1_024, 512)?,
        qk_gamma: load_tensor(
            file,
            LABEL,
            &format!("{prefix}.qk_offset_scale.gamma"),
            &[4, QUERY_KEY_DIM],
        )?,
        qk_beta: load_tensor(
            file,
            LABEL,
            &format!("{prefix}.qk_offset_scale.beta"),
            &[4, QUERY_KEY_DIM],
        )?,
    })
}

fn load_fsmn(file: &GgufFile, prefix: &str) -> Result<GatedFsmnLayer> {
    let core = format!("{prefix}.gated_fsmn");
    Ok(GatedFsmnLayer {
        input_conv: load_conv(file, &format!("{prefix}.conv1.0"), 512, 256, 1, 1, true)?,
        input_slope: load_scalar(file, &format!("{prefix}.conv1.1.weight"))?,
        norm1: load_norm(file, &format!("{prefix}.norm1"), FSMN_CHANNELS)?,
        to_u: load_ff_layer(file, &format!("{core}.to_u"), FSMN_CHANNELS)?,
        to_v: load_ff_layer(file, &format!("{core}.to_v"), FSMN_CHANNELS)?,
        fsmn: FsmnCore {
            linear: load_linear(
                file,
                &format!("{core}.fsmn.linear"),
                FSMN_CHANNELS,
                FSMN_CHANNELS,
                true,
            )?,
            project: load_linear(
                file,
                &format!("{core}.fsmn.project"),
                FSMN_CHANNELS,
                FSMN_CHANNELS,
                false,
            )?,
            dense: [
                DenseStage {
                    conv: load_conv4_grouped(
                        file,
                        &format!("{core}.fsmn.conv.conv1"),
                        256,
                        256,
                        39,
                        256,
                    )?,
                    norm: load_norm(file, &format!("{core}.fsmn.conv.norm1"), 256)?,
                    slope: load_tensor(
                        file,
                        LABEL,
                        &format!("{core}.fsmn.conv.prelu1.weight"),
                        &[256],
                    )?,
                    dilation: 1,
                },
                DenseStage {
                    conv: load_conv4_grouped(
                        file,
                        &format!("{core}.fsmn.conv.conv2"),
                        512,
                        256,
                        39,
                        256,
                    )?,
                    norm: load_norm(file, &format!("{core}.fsmn.conv.norm2"), 256)?,
                    slope: load_tensor(
                        file,
                        LABEL,
                        &format!("{core}.fsmn.conv.prelu2.weight"),
                        &[256],
                    )?,
                    dilation: 2,
                },
            ],
        },
        norm2: load_norm(file, &format!("{prefix}.norm2"), FSMN_CHANNELS)?,
        output_conv: load_conv(file, &format!("{prefix}.conv2"), 256, 512, 1, 1, true)?,
    })
}

fn load_ff_scale(file: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<FfConv> {
    Ok(FfConv {
        norm: FfNorm::Scale {
            gain: load_scalar(file, &format!("{prefix}.mdl.0.g"))?,
        },
        linear: load_linear(file, &format!("{prefix}.mdl.1"), input, output, true)?,
        depthwise: load_conv(
            file,
            &format!("{prefix}.mdl.3.sequential.1.conv"),
            output,
            output,
            17,
            output,
            false,
        )?,
    })
}

fn load_ff_layer(file: &GgufFile, prefix: &str, channels: usize) -> Result<FfConv> {
    Ok(FfConv {
        norm: FfNorm::Layer(load_norm(file, &format!("{prefix}.mdl.0"), channels)?),
        linear: load_linear(file, &format!("{prefix}.mdl.1"), channels, channels, true)?,
        depthwise: load_conv(
            file,
            &format!("{prefix}.mdl.3.sequential.1.conv"),
            channels,
            channels,
            17,
            channels,
            false,
        )?,
    })
}

fn load_norm(file: &GgufFile, prefix: &str, channels: usize) -> Result<Norm> {
    Ok(Norm {
        gamma: load_tensor(file, LABEL, &format!("{prefix}.weight"), &[channels])?,
        beta: load_tensor(file, LABEL, &format!("{prefix}.bias"), &[channels])?,
    })
}

fn load_linear(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<Linear> {
    let weight = load_tensor(file, LABEL, &format!("{prefix}.weight"), &[output, input])?;
    let bias = bias
        .then(|| load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output]))
        .transpose()?;
    Ok(Linear {
        weight_t: transpose_out_in(weight, input, output),
        bias,
        input,
        output,
    })
}

fn load_conv(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    groups: usize,
    bias: bool,
) -> Result<Conv1d> {
    let weight = load_tensor(
        file,
        LABEL,
        &format!("{prefix}.weight"),
        &[output, input / groups, kernel],
    )?;
    let bias = bias
        .then(|| load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output]))
        .transpose()?;
    Ok(Conv1d {
        weight,
        bias,
        input,
        output,
        kernel,
        groups,
    })
}

fn load_conv4_grouped(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    groups: usize,
) -> Result<Conv1d> {
    let weight = load_tensor(
        file,
        LABEL,
        &format!("{prefix}.weight"),
        &[output, input / groups, kernel, 1],
    )?;
    Ok(Conv1d {
        weight,
        bias: None,
        input,
        output,
        kernel,
        groups,
    })
}

fn load_scalar(file: &GgufFile, name: &str) -> Result<f32> {
    Ok(load_tensor(file, LABEL, name, &[1])?[0])
}

fn transpose_out_in(weight: Vec<f32>, input: usize, output: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; weight.len()];
    for out in 0..output {
        for inner in 0..input {
            transposed[inner * output + out] = weight[out * input + inner];
        }
    }
    transposed
}
