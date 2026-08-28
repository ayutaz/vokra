use vokra_core::gguf::{GgmlType, GgufFile};
use vokra_core::{Result, VokraError};

use crate::strict_checkpoint::load_tensor;

use super::{DENSE_CHANNELS, GRU_HIDDEN, TS_BLOCKS};

const LABEL: &str = "mp_senet";

#[derive(Debug, Clone)]
pub(super) struct Conv2d {
    pub(super) weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) kernel_h: usize,
    pub(super) kernel_w: usize,
}

#[derive(Debug, Clone)]
pub(super) struct Norm {
    pub(super) gamma: Vec<f32>,
    pub(super) beta: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct ConvNormPrelu {
    pub(super) conv: Conv2d,
    pub(super) norm: Norm,
    pub(super) slope: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct DenseLayer {
    pub(super) conv: Conv2d,
    pub(super) norm: Norm,
    pub(super) slope: Vec<f32>,
    pub(super) dilation_h: usize,
}

#[derive(Debug, Clone)]
pub(super) struct DenseBlock {
    pub(super) layers: Vec<DenseLayer>,
}

#[derive(Debug, Clone)]
pub(super) struct DenseEncoder {
    pub(super) input: ConvNormPrelu,
    pub(super) dense: DenseBlock,
    pub(super) downsample: ConvNormPrelu,
}

#[derive(Debug, Clone)]
pub(super) struct DecoderStem {
    pub(super) expand: Conv2d,
    pub(super) norm: Norm,
    pub(super) slope: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct MaskDecoder {
    pub(super) dense: DenseBlock,
    pub(super) stem: DecoderStem,
    pub(super) output: Conv2d,
    pub(super) sigmoid_slope: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct PhaseDecoder {
    pub(super) dense: DenseBlock,
    pub(super) stem: DecoderStem,
    pub(super) real: Conv2d,
    pub(super) imag: Conv2d,
}

#[derive(Debug, Clone)]
pub(super) struct Attention {
    /// PyTorch `[3 * dim, dim]` transposed to Compute `[dim, 3 * dim]`.
    pub(super) in_weight_t: Vec<f32>,
    /// Q slice of `in_weight_t`, used by the device-resident attention path.
    pub(super) q_weight_t: Vec<f32>,
    pub(super) in_bias: Vec<f32>,
    /// PyTorch `[dim, dim]` transposed to Compute `[dim, dim]`.
    pub(super) out_weight_t: Vec<f32>,
    pub(super) out_bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct GruDirection {
    /// PyTorch `[3 * hidden, input]` transposed for row-major GEMM.
    pub(super) weight_ih_t: Vec<f32>,
    /// PyTorch `[3 * hidden, hidden]` transposed for row-major GEMM.
    pub(super) weight_hh_t: Vec<f32>,
    pub(super) bias_ih: Vec<f32>,
    pub(super) bias_hh: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct BiGru {
    pub(super) forward: GruDirection,
    pub(super) reverse: GruDirection,
}

#[derive(Debug, Clone)]
pub(super) struct Linear {
    /// PyTorch `[output, input]` transposed to Compute `[input, output]`.
    pub(super) weight_t: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) input: usize,
    pub(super) output: usize,
}

#[derive(Debug, Clone)]
pub(super) struct Transformer {
    pub(super) norm1: Norm,
    pub(super) attention: Attention,
    pub(super) norm2: Norm,
    pub(super) gru: BiGru,
    pub(super) linear: Linear,
    pub(super) norm3: Norm,
}

#[derive(Debug, Clone)]
pub(super) struct TsBlock {
    pub(super) time: Transformer,
    pub(super) frequency: Transformer,
}

#[derive(Debug, Clone)]
pub(super) struct MpSenetWeights {
    pub(super) encoder: DenseEncoder,
    pub(super) transformers: Vec<TsBlock>,
    pub(super) mask: MaskDecoder,
    pub(super) phase: PhaseDecoder,
}

impl MpSenetWeights {
    pub(super) fn bind(file: &GgufFile) -> Result<Self> {
        let encoder = DenseEncoder {
            input: load_conv_norm_prelu(file, "dense_encoder.dense_conv_1", 2, 64, 1, 1)?,
            dense: load_dense_block(file, "dense_encoder.dense_block")?,
            downsample: load_conv_norm_prelu(file, "dense_encoder.dense_conv_2", 64, 64, 1, 3)?,
        };
        let mut transformers = Vec::with_capacity(TS_BLOCKS);
        for block in 0..TS_BLOCKS {
            let prefix = format!("TSTransformer.{block}");
            transformers.push(TsBlock {
                time: load_transformer(file, &format!("{prefix}.time_transformer"))?,
                frequency: load_transformer(file, &format!("{prefix}.freq_transformer"))?,
            });
        }
        Ok(Self {
            encoder,
            transformers,
            mask: MaskDecoder {
                dense: load_dense_block(file, "mask_decoder.dense_block")?,
                stem: load_decoder_stem(file, "mask_decoder.mask_conv")?,
                output: load_conv(file, "mask_decoder.mask_conv.3", 64, 1, 1, 2)?,
                sigmoid_slope: load_finite(file, "mask_decoder.lsigmoid.slope", &[201, 1])?,
            },
            phase: PhaseDecoder {
                dense: load_dense_block(file, "phase_decoder.dense_block")?,
                stem: load_decoder_stem(file, "phase_decoder.phase_conv")?,
                real: load_conv(file, "phase_decoder.phase_conv_r", 64, 1, 1, 2)?,
                imag: load_conv(file, "phase_decoder.phase_conv_i", 64, 1, 1, 2)?,
            },
        })
    }
}

fn load_transformer(file: &GgufFile, prefix: &str) -> Result<Transformer> {
    let attention_prefix = format!("{prefix}.attention");
    let in_weight = load_finite(
        file,
        &format!("{attention_prefix}.in_proj_weight"),
        &[3 * DENSE_CHANNELS, DENSE_CHANNELS],
    )?;
    let out_weight = load_finite(
        file,
        &format!("{attention_prefix}.out_proj.weight"),
        &[DENSE_CHANNELS, DENSE_CHANNELS],
    )?;
    let in_weight_t = transpose_out_in(&in_weight, 3 * DENSE_CHANNELS, DENSE_CHANNELS);
    let mut q_weight_t = vec![0.0f32; DENSE_CHANNELS * DENSE_CHANNELS];
    for input in 0..DENSE_CHANNELS {
        q_weight_t[input * DENSE_CHANNELS..(input + 1) * DENSE_CHANNELS].copy_from_slice(
            &in_weight_t[input * 3 * DENSE_CHANNELS..input * 3 * DENSE_CHANNELS + DENSE_CHANNELS],
        );
    }
    Ok(Transformer {
        norm1: load_norm(file, &format!("{prefix}.norm1"), DENSE_CHANNELS)?,
        attention: Attention {
            in_weight_t,
            q_weight_t,
            in_bias: load_finite(
                file,
                &format!("{attention_prefix}.in_proj_bias"),
                &[3 * DENSE_CHANNELS],
            )?,
            out_weight_t: transpose_out_in(&out_weight, DENSE_CHANNELS, DENSE_CHANNELS),
            out_bias: load_finite(
                file,
                &format!("{attention_prefix}.out_proj.bias"),
                &[DENSE_CHANNELS],
            )?,
        },
        norm2: load_norm(file, &format!("{prefix}.norm2"), DENSE_CHANNELS)?,
        gru: BiGru {
            forward: load_gru_direction(file, &format!("{prefix}.ffn.gru"), "")?,
            reverse: load_gru_direction(file, &format!("{prefix}.ffn.gru"), "_reverse")?,
        },
        linear: load_linear(
            file,
            &format!("{prefix}.ffn.linear"),
            2 * GRU_HIDDEN,
            DENSE_CHANNELS,
        )?,
        norm3: load_norm(file, &format!("{prefix}.norm3"), DENSE_CHANNELS)?,
    })
}

fn load_gru_direction(file: &GgufFile, prefix: &str, suffix: &str) -> Result<GruDirection> {
    let weight_ih = load_finite(
        file,
        &format!("{prefix}.weight_ih_l0{suffix}"),
        &[3 * GRU_HIDDEN, DENSE_CHANNELS],
    )?;
    let weight_hh = load_finite(
        file,
        &format!("{prefix}.weight_hh_l0{suffix}"),
        &[3 * GRU_HIDDEN, GRU_HIDDEN],
    )?;
    Ok(GruDirection {
        weight_ih_t: transpose_out_in(&weight_ih, 3 * GRU_HIDDEN, DENSE_CHANNELS),
        weight_hh_t: transpose_out_in(&weight_hh, 3 * GRU_HIDDEN, GRU_HIDDEN),
        bias_ih: load_finite(
            file,
            &format!("{prefix}.bias_ih_l0{suffix}"),
            &[3 * GRU_HIDDEN],
        )?,
        bias_hh: load_finite(
            file,
            &format!("{prefix}.bias_hh_l0{suffix}"),
            &[3 * GRU_HIDDEN],
        )?,
    })
}

fn load_dense_block(file: &GgufFile, prefix: &str) -> Result<DenseBlock> {
    let mut layers = Vec::with_capacity(4);
    for layer in 0..4 {
        let root = format!("{prefix}.dense_block.{layer}");
        layers.push(DenseLayer {
            conv: load_conv(
                file,
                &format!("{root}.1"),
                DENSE_CHANNELS * (layer + 1),
                DENSE_CHANNELS,
                2,
                3,
            )?,
            norm: load_norm(file, &format!("{root}.2"), DENSE_CHANNELS)?,
            slope: load_finite(file, &format!("{root}.3.weight"), &[DENSE_CHANNELS])?,
            dilation_h: 1 << layer,
        });
    }
    Ok(DenseBlock { layers })
}

fn load_conv_norm_prelu(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> Result<ConvNormPrelu> {
    Ok(ConvNormPrelu {
        conv: load_conv(
            file,
            &format!("{prefix}.0"),
            input,
            output,
            kernel_h,
            kernel_w,
        )?,
        norm: load_norm(file, &format!("{prefix}.1"), output)?,
        slope: load_finite(file, &format!("{prefix}.2.weight"), &[output])?,
    })
}

fn load_decoder_stem(file: &GgufFile, prefix: &str) -> Result<DecoderStem> {
    Ok(DecoderStem {
        expand: load_conv(file, &format!("{prefix}.0.conv"), 64, 128, 1, 3)?,
        norm: load_norm(file, &format!("{prefix}.1"), 64)?,
        slope: load_finite(file, &format!("{prefix}.2.weight"), &[64])?,
    })
}

fn load_conv(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> Result<Conv2d> {
    Ok(Conv2d {
        weight: load_finite(
            file,
            &format!("{prefix}.weight"),
            &[output, input, kernel_h, kernel_w],
        )?,
        bias: load_finite(file, &format!("{prefix}.bias"), &[output])?,
        input,
        output,
        kernel_h,
        kernel_w,
    })
}

fn load_norm(file: &GgufFile, prefix: &str, channels: usize) -> Result<Norm> {
    Ok(Norm {
        gamma: load_finite(file, &format!("{prefix}.weight"), &[channels])?,
        beta: load_finite(file, &format!("{prefix}.bias"), &[channels])?,
    })
}

fn load_linear(file: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<Linear> {
    let weight = load_finite(file, &format!("{prefix}.weight"), &[output, input])?;
    Ok(Linear {
        weight_t: transpose_out_in(&weight, output, input),
        bias: load_finite(file, &format!("{prefix}.bias"), &[output])?,
        input,
        output,
    })
}

fn load_finite(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("{LABEL}: required tensor `{name}` is missing"))
    })?;
    if info.dtype != GgmlType::F32 {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{name}` uses {:?}; the audited release is F32 and no quantized MP-SENet parity has been established",
            info.dtype
        )));
    }
    let values = load_tensor(file, LABEL, name, expected)?;
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{name}` contains a non-finite value at element {index}"
        )));
    }
    Ok(values)
}

fn transpose_out_in(weight: &[f32], output: usize, input: usize) -> Vec<f32> {
    debug_assert_eq!(weight.len(), output * input);
    let mut transposed = vec![0.0f32; weight.len()];
    for out in 0..output {
        for inner in 0..input {
            transposed[inner * output + out] = weight[out * input + inner];
        }
    }
    transposed
}
