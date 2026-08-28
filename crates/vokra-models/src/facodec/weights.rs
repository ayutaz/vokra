use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};
use vokra_ops::mimi_rvq::CodebookTable;
use vokra_ops::{DacOutProj, DacRvqAttrs};

use crate::strict_checkpoint::load_tensor;

use super::{CODEBOOK_DIM, CODEBOOK_SIZE, DIM, LABEL};

pub(super) const TRANSFORMER_LAYERS: usize = 4;
pub(super) const TRANSFORMER_HEADS: usize = 4;
pub(super) const TRANSFORMER_FFN: usize = 1_024;
pub(super) const POSITION_LIMIT: usize = 5_000;

#[derive(Debug, Clone)]
pub(super) struct Linear {
    pub(super) input: usize,
    pub(super) output: usize,
    /// Row-major `[input, output]`, ready for `Compute::gemm_f32`.
    pub(super) weight_t: Vec<f32>,
    pub(super) bias: Vec<f32>,
}

impl Linear {
    pub(super) fn bind(file: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<Self> {
        let weight = load_tensor(file, LABEL, &format!("{prefix}.weight"), &[output, input])?;
        Ok(Self {
            input,
            output,
            weight_t: transpose_linear(&weight, input, output),
            bias: load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output])?,
        })
    }

    pub(super) fn bind_weight_norm(
        file: &GgufFile,
        prefix: &str,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        let magnitude = load_tensor(file, LABEL, &format!("{prefix}.weight_g"), &[output, 1])?;
        let direction = load_tensor(file, LABEL, &format!("{prefix}.weight_v"), &[output, input])?;
        let folded = fold_weight_norm_rows(&magnitude, &direction, input, output, prefix)?;
        Ok(Self {
            input,
            output,
            weight_t: transpose_linear(&folded, input, output),
            bias: load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output])?,
        })
    }

    fn bind_multihead_attention(file: &GgufFile, prefix: &str) -> Result<Self> {
        let output = 3 * DIM;
        let weight = load_tensor(
            file,
            LABEL,
            &format!("{prefix}.in_proj_weight"),
            &[output, DIM],
        )?;
        Ok(Self {
            input: DIM,
            output,
            weight_t: transpose_linear(&weight, DIM, output),
            bias: load_tensor(file, LABEL, &format!("{prefix}.in_proj_bias"), &[output])?,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct Conv1d {
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) kernel: usize,
    pub(super) stride: usize,
    pub(super) padding: usize,
    pub(super) dilation: usize,
    /// Folded PyTorch layout `[output, input, kernel]`.
    pub(super) weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
}

impl Conv1d {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn bind(
        file: &GgufFile,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self> {
        let magnitude = load_tensor(file, LABEL, &format!("{prefix}.weight_g"), &[output, 1, 1])?;
        let direction = load_tensor(
            file,
            LABEL,
            &format!("{prefix}.weight_v"),
            &[output, input, kernel],
        )?;
        let weight = fold_weight_norm_rows(&magnitude, &direction, input * kernel, output, prefix)?;
        Ok(Self {
            input,
            output,
            kernel,
            stride,
            padding,
            dilation,
            weight,
            bias: load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output])?,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct ConvTranspose1d {
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) kernel: usize,
    pub(super) stride: usize,
    pub(super) padding: usize,
    pub(super) output_padding: usize,
    /// Folded PyTorch layout `[input, output, kernel]`.
    pub(super) weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
}

impl ConvTranspose1d {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn bind(
        file: &GgufFile,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> Result<Self> {
        // torch.nn.utils.weight_norm defaults to dim=0. ConvTranspose1d stores
        // `[input, output, kernel]`, so the norm is one row per input channel.
        let magnitude = load_tensor(file, LABEL, &format!("{prefix}.weight_g"), &[input, 1, 1])?;
        let direction = load_tensor(
            file,
            LABEL,
            &format!("{prefix}.weight_v"),
            &[input, output, kernel],
        )?;
        let weight = fold_weight_norm_rows(&magnitude, &direction, output * kernel, input, prefix)?;
        Ok(Self {
            input,
            output,
            kernel,
            stride,
            padding,
            output_padding,
            weight,
            bias: load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output])?,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct AliasFreeActivation {
    pub(super) alpha: Vec<f32>,
    pub(super) beta: Vec<f32>,
    pub(super) upsample_filter: Vec<f32>,
    pub(super) downsample_filter: Vec<f32>,
}

impl AliasFreeActivation {
    pub(super) fn bind(file: &GgufFile, prefix: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            alpha: load_tensor(file, LABEL, &format!("{prefix}.act.alpha"), &[channels])?,
            beta: load_tensor(file, LABEL, &format!("{prefix}.act.beta"), &[channels])?,
            upsample_filter: load_tensor(
                file,
                LABEL,
                &format!("{prefix}.upsample.filter"),
                &[1, 1, 12],
            )?,
            downsample_filter: load_tensor(
                file,
                LABEL,
                &format!("{prefix}.downsample.lowpass.filter"),
                &[1, 1, 12],
            )?,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct ResidualUnit {
    pub(super) first_activation: AliasFreeActivation,
    pub(super) first_conv: Conv1d,
    pub(super) second_activation: AliasFreeActivation,
    pub(super) second_conv: Conv1d,
}

impl ResidualUnit {
    pub(super) fn bind(
        file: &GgufFile,
        prefix: &str,
        channels: usize,
        dilation: usize,
    ) -> Result<Self> {
        Ok(Self {
            first_activation: AliasFreeActivation::bind(
                file,
                &format!("{prefix}.block.0"),
                channels,
            )?,
            first_conv: Conv1d::bind(
                file,
                &format!("{prefix}.block.1"),
                channels,
                channels,
                7,
                1,
                3 * dilation,
                dilation,
            )?,
            second_activation: AliasFreeActivation::bind(
                file,
                &format!("{prefix}.block.2"),
                channels,
            )?,
            second_conv: Conv1d::bind(
                file,
                &format!("{prefix}.block.3"),
                channels,
                channels,
                1,
                1,
                0,
                1,
            )?,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct EncoderStage {
    pub(super) residuals: Vec<ResidualUnit>,
    pub(super) activation: AliasFreeActivation,
    pub(super) downsample: Conv1d,
}

#[derive(Debug, Clone)]
pub(super) struct EncoderWeights {
    pub(super) pre: Conv1d,
    pub(super) stages: Vec<EncoderStage>,
    pub(super) post_activation: AliasFreeActivation,
    pub(super) post: Conv1d,
    pub(super) hann_window: Vec<f32>,
    pub(super) mel_basis: Vec<f32>,
}

impl EncoderWeights {
    fn bind(file: &GgufFile) -> Result<Self> {
        let pre = Conv1d::bind(file, "encoder.block.0", 1, 32, 7, 1, 3, 1)?;
        let strides = [2usize, 4, 5, 5];
        let mut input = 32usize;
        let mut stages = Vec::with_capacity(strides.len());
        for (zero_index, stride) in strides.into_iter().enumerate() {
            let module_index = zero_index + 1;
            let prefix = format!("encoder.block.{module_index}");
            let residuals = [1usize, 3, 9]
                .into_iter()
                .enumerate()
                .map(|(unit, dilation)| {
                    ResidualUnit::bind(file, &format!("{prefix}.block.{unit}"), input, dilation)
                })
                .collect::<Result<Vec<_>>>()?;
            let output = input * 2;
            stages.push(EncoderStage {
                residuals,
                activation: AliasFreeActivation::bind(file, &format!("{prefix}.block.3"), input)?,
                downsample: Conv1d::bind(
                    file,
                    &format!("{prefix}.block.4"),
                    input,
                    output,
                    2 * stride,
                    stride,
                    stride / 2 + stride % 2,
                    1,
                )?,
            });
            input = output;
        }
        Ok(Self {
            pre,
            stages,
            post_activation: AliasFreeActivation::bind(file, "encoder.block.5", 512)?,
            post: Conv1d::bind(file, "encoder.block.6", 512, DIM, 3, 1, 1, 1)?,
            hann_window: load_tensor(file, LABEL, "encoder.mel_transform.hann_window", &[800])?,
            mel_basis: load_tensor(file, LABEL, "encoder.mel_transform.mel_basis", &[80, 513])?,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct TransformerLayer {
    pub(super) norm1_weight: Vec<f32>,
    pub(super) norm1_bias: Vec<f32>,
    pub(super) attention_in: Linear,
    pub(super) attention_out: Linear,
    pub(super) norm2_weight: Vec<f32>,
    pub(super) norm2_bias: Vec<f32>,
    pub(super) ffn_conv_weight: Vec<f32>,
    pub(super) ffn_conv_bias: Vec<f32>,
    pub(super) ffn_out: Linear,
}

#[derive(Debug, Clone)]
pub(super) struct TransformerWeights {
    pub(super) position: Vec<f32>,
    pub(super) layers: Vec<TransformerLayer>,
    pub(super) last_norm_weight: Vec<f32>,
    pub(super) last_norm_bias: Vec<f32>,
}

impl TransformerWeights {
    fn bind(file: &GgufFile, prefix: &str) -> Result<Self> {
        let mut layers = Vec::with_capacity(TRANSFORMER_LAYERS);
        for index in 0..TRANSFORMER_LAYERS {
            let layer = format!("{prefix}.layers.{index}");
            layers.push(TransformerLayer {
                norm1_weight: load_tensor(file, LABEL, &format!("{layer}.ln_1.weight"), &[DIM])?,
                norm1_bias: load_tensor(file, LABEL, &format!("{layer}.ln_1.bias"), &[DIM])?,
                attention_in: Linear::bind_multihead_attention(
                    file,
                    &format!("{layer}.self_attn"),
                )?,
                attention_out: Linear::bind(
                    file,
                    &format!("{layer}.self_attn.out_proj"),
                    DIM,
                    DIM,
                )?,
                norm2_weight: load_tensor(file, LABEL, &format!("{layer}.ln_2.weight"), &[DIM])?,
                norm2_bias: load_tensor(file, LABEL, &format!("{layer}.ln_2.bias"), &[DIM])?,
                ffn_conv_weight: load_tensor(
                    file,
                    LABEL,
                    &format!("{layer}.ffn.ffn_1.weight"),
                    &[TRANSFORMER_FFN, DIM, 5],
                )?,
                ffn_conv_bias: load_tensor(
                    file,
                    LABEL,
                    &format!("{layer}.ffn.ffn_1.bias"),
                    &[TRANSFORMER_FFN],
                )?,
                ffn_out: Linear::bind(file, &format!("{layer}.ffn.ffn_2"), TRANSFORMER_FFN, DIM)?,
            });
        }
        Ok(Self {
            position: load_tensor(
                file,
                LABEL,
                &format!("{prefix}.position_emb.pe"),
                &[POSITION_LIMIT, 1, DIM],
            )?,
            layers,
            last_norm_weight: load_tensor(
                file,
                LABEL,
                &format!("{prefix}.last_ln.weight"),
                &[DIM],
            )?,
            last_norm_bias: load_tensor(file, LABEL, &format!("{prefix}.last_ln.bias"), &[DIM])?,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct QuantizerLayer {
    pub(super) input_projection: Linear,
    pub(super) output_projection: Linear,
    pub(super) codebook: Vec<f32>,
    pub(super) normalized_codebook_t: Vec<f32>,
    pub(super) normalized_codebook_norm2: Vec<f32>,
}

impl QuantizerLayer {
    fn bind(file: &GgufFile, prefix: &str) -> Result<Self> {
        let codebook = load_tensor(
            file,
            LABEL,
            &format!("{prefix}._codebook.weight"),
            &[CODEBOOK_SIZE, CODEBOOK_DIM],
        )?;
        let mut normalized_codebook_t = vec![0.0f32; CODEBOOK_DIM * CODEBOOK_SIZE];
        let mut normalized_codebook_norm2 = vec![0.0f32; CODEBOOK_SIZE];
        for row in 0..CODEBOOK_SIZE {
            let source = &codebook[row * CODEBOOK_DIM..(row + 1) * CODEBOOK_DIM];
            let norm = source.iter().map(|value| value * value).sum::<f32>().sqrt();
            if !norm.is_finite() {
                return Err(VokraError::ModelLoad(format!(
                    "{LABEL}: `{prefix}._codebook.weight` row {row} has invalid L2 norm {norm}"
                )));
            }
            // torch.nn.functional.normalize uses max(norm, eps), with its
            // default eps=1e-12. Preserve zero rows rather than turning an
            // otherwise valid exact checkpoint into a runtime-only error.
            let denominator = norm.max(1.0e-12);
            for column in 0..CODEBOOK_DIM {
                let value = source[column] / denominator;
                normalized_codebook_t[column * CODEBOOK_SIZE + row] = value;
                normalized_codebook_norm2[row] += value * value;
            }
        }
        Ok(Self {
            input_projection: Linear::bind_weight_norm(
                file,
                &format!("{prefix}.in_proj"),
                DIM,
                CODEBOOK_DIM,
            )?,
            output_projection: Linear::bind_weight_norm(
                file,
                &format!("{prefix}.out_proj"),
                CODEBOOK_DIM,
                DIM,
            )?,
            codebook,
            normalized_codebook_t,
            normalized_codebook_norm2,
        })
    }

    fn codebook_table(&self) -> Result<CodebookTable> {
        CodebookTable::new(CODEBOOK_SIZE, CODEBOOK_DIM, self.codebook.clone())
    }

    fn out_projection(&self) -> Result<DacOutProj> {
        let mut weight = vec![0.0f32; DIM * CODEBOOK_DIM];
        for input in 0..CODEBOOK_DIM {
            for output in 0..DIM {
                weight[output * CODEBOOK_DIM + input] =
                    self.output_projection.weight_t[input * DIM + output];
            }
        }
        DacOutProj::new(
            DIM,
            CODEBOOK_DIM,
            weight,
            self.output_projection.bias.clone(),
        )
    }
}

#[derive(Debug, Clone)]
pub(super) struct QuantizerWeights {
    pub(super) groups: [Vec<QuantizerLayer>; 3],
    pub(super) codebook_tables: Vec<CodebookTable>,
    pub(super) output_projections: Vec<DacOutProj>,
    pub(super) attrs: DacRvqAttrs,
}

impl QuantizerWeights {
    fn bind(file: &GgufFile) -> Result<Self> {
        let bind_group = |group: usize, layers: usize| {
            (0..layers)
                .map(|layer| {
                    QuantizerLayer::bind(file, &format!("decoder.quantizer.{group}.layers.{layer}"))
                })
                .collect::<Result<Vec<_>>>()
        };
        let groups = [bind_group(0, 1)?, bind_group(1, 2)?, bind_group(2, 3)?];
        let codebook_tables = groups
            .iter()
            .flatten()
            .map(QuantizerLayer::codebook_table)
            .collect::<Result<Vec<_>>>()?;
        let output_projections = groups
            .iter()
            .flatten()
            .map(QuantizerLayer::out_projection)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            groups,
            codebook_tables,
            output_projections,
            attrs: DacRvqAttrs {
                n_codebooks: 6,
                codebook_size: CODEBOOK_SIZE,
                codebook_dim: CODEBOOK_DIM,
                d_model: DIM,
            },
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct DecoderStage {
    pub(super) activation: AliasFreeActivation,
    pub(super) upsample: ConvTranspose1d,
    pub(super) residuals: Vec<ResidualUnit>,
}

#[derive(Debug, Clone)]
pub(super) struct DecoderWeights {
    pub(super) quantizer: QuantizerWeights,
    pub(super) timbre_encoder: TransformerWeights,
    pub(super) timbre_linear: Linear,
    pub(super) melspec_linear: Linear,
    pub(super) melspec_encoder: TransformerWeights,
    pub(super) pre: Conv1d,
    pub(super) stages: Vec<DecoderStage>,
    pub(super) post_activation: AliasFreeActivation,
    pub(super) post: Conv1d,
}

impl DecoderWeights {
    fn bind(file: &GgufFile) -> Result<Self> {
        let strides = [5usize, 5, 4, 2];
        let mut input = 1_024usize;
        let mut stages = Vec::with_capacity(strides.len());
        for (zero_index, stride) in strides.into_iter().enumerate() {
            let module_index = zero_index + 1;
            let prefix = format!("decoder.model.{module_index}");
            let output = input / 2;
            stages.push(DecoderStage {
                activation: AliasFreeActivation::bind(file, &format!("{prefix}.block.0"), input)?,
                upsample: ConvTranspose1d::bind(
                    file,
                    &format!("{prefix}.block.1"),
                    input,
                    output,
                    2 * stride,
                    stride,
                    stride / 2 + stride % 2,
                    stride % 2,
                )?,
                residuals: [1usize, 3, 9]
                    .into_iter()
                    .enumerate()
                    .map(|(unit, dilation)| {
                        ResidualUnit::bind(
                            file,
                            &format!("{prefix}.block.{}", unit + 2),
                            output,
                            dilation,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            });
            input = output;
        }
        Ok(Self {
            quantizer: QuantizerWeights::bind(file)?,
            timbre_encoder: TransformerWeights::bind(file, "decoder.timbre_encoder")?,
            timbre_linear: Linear::bind(file, "decoder.timbre_linear", DIM, 2 * DIM)?,
            melspec_linear: Linear::bind(file, "decoder.melspec_linear", 20, DIM)?,
            melspec_encoder: TransformerWeights::bind(file, "decoder.melspec_encoder")?,
            pre: Conv1d::bind(file, "decoder.model.0", DIM, 1_024, 7, 1, 3, 1)?,
            stages,
            post_activation: AliasFreeActivation::bind(file, "decoder.model.5", 64)?,
            post: Conv1d::bind(file, "decoder.model.6", 64, 1, 7, 1, 3, 1)?,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct FacodecWeights {
    pub(super) encoder: EncoderWeights,
    pub(super) decoder: DecoderWeights,
}

impl FacodecWeights {
    pub(super) fn bind(file: &GgufFile) -> Result<Self> {
        Ok(Self {
            encoder: EncoderWeights::bind(file)?,
            decoder: DecoderWeights::bind(file)?,
        })
    }
}

fn transpose_linear(weight: &[f32], input: usize, output: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; weight.len()];
    for out in 0..output {
        for inner in 0..input {
            transposed[inner * output + out] = weight[out * input + inner];
        }
    }
    transposed
}

fn fold_weight_norm_rows(
    magnitude: &[f32],
    direction: &[f32],
    row_width: usize,
    rows: usize,
    label: &str,
) -> Result<Vec<f32>> {
    if magnitude.len() != rows || direction.len() != rows * row_width {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{label}` weight_norm buffers have g={} v={}, expected {rows} and {}",
            magnitude.len(),
            direction.len(),
            rows * row_width
        )));
    }
    let mut folded = vec![0.0f32; direction.len()];
    for row in 0..rows {
        let source = &direction[row * row_width..(row + 1) * row_width];
        let norm = source.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: `{label}` weight_norm row {row} has invalid norm {norm}"
            )));
        }
        let scale = magnitude[row] / norm;
        for column in 0..row_width {
            folded[row * row_width + column] = source[column] * scale;
        }
    }
    Ok(folded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_norm_folds_rows_independently() {
        let folded =
            fold_weight_norm_rows(&[5.0, 13.0], &[3.0, 4.0, 5.0, 12.0], 2, 2, "test").unwrap();
        assert_eq!(folded, vec![3.0, 4.0, 5.0, 12.0]);
    }
}
