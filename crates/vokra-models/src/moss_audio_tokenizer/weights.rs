use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::Compute;
use crate::strict_checkpoint::load_tensor;

pub(super) const CODEBOOK_SIZE: usize = 1_024;
pub(super) const CODEBOOK_DIM: usize = 8;
pub(super) const RVQ_DIM: usize = 512;
pub(super) const LATENT_DIM: usize = 768;
pub(super) const MODEL_DIM: usize = 256;
pub(super) const FFN_DIM: usize = 1_024;
pub(super) const NUM_HEADS: usize = 4;
pub(super) const NUM_QUANTIZERS: usize = 16;
const LABEL: &str = "moss_audio_tokenizer/nano";

#[derive(Debug, Clone, Copy)]
pub(super) struct StageSpec {
    pub(super) module_index: usize,
    pub(super) input_dim: usize,
    pub(super) output_dim: usize,
    pub(super) layers: usize,
    pub(super) context: usize,
}

pub(super) const STAGE_SPECS: [StageSpec; 4] = [
    StageSpec {
        module_index: 1,
        input_dim: 192,
        output_dim: 768,
        layers: 4,
        context: 500,
    },
    StageSpec {
        module_index: 3,
        input_dim: 384,
        output_dim: 768,
        layers: 2,
        context: 800,
    },
    StageSpec {
        module_index: 5,
        input_dim: 384,
        output_dim: 768,
        layers: 2,
        context: 1_200,
    },
    StageSpec {
        module_index: 7,
        input_dim: 384,
        output_dim: 240,
        layers: 4,
        context: 1_600,
    },
];

#[derive(Debug, Clone)]
pub(super) struct Linear {
    pub(super) input: usize,
    pub(super) output: usize,
    /// Row-major `[input, output]`, ready for `Compute::gemm_f32`.
    weight_t: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl Linear {
    fn load(
        file: &GgufFile,
        name: &str,
        input: usize,
        output: usize,
        bias_name: Option<&str>,
    ) -> Result<Self> {
        let weight = load_tensor(file, LABEL, name, &[output, input])?;
        let bias = bias_name
            .map(|name| load_tensor(file, LABEL, name, &[output]))
            .transpose()?;
        Ok(Self {
            input,
            output,
            weight_t: transpose_pytorch_linear(&weight, input, output),
            bias,
        })
    }

    fn load_weight_norm_1x1(
        file: &GgufFile,
        prefix: &str,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        let magnitude = load_tensor(
            file,
            LABEL,
            &format!("{prefix}.parametrizations.weight.original0"),
            &[output, 1, 1],
        )?;
        let direction = load_tensor(
            file,
            LABEL,
            &format!("{prefix}.parametrizations.weight.original1"),
            &[output, input, 1],
        )?;
        let weight = fold_weight_norm_rows(&magnitude, &direction, input, output, prefix)?;
        Ok(Self {
            input,
            output,
            weight_t: transpose_pytorch_linear(&weight, input, output),
            bias: Some(load_tensor(
                file,
                LABEL,
                &format!("{prefix}.bias"),
                &[output],
            )?),
        })
    }

    pub(super) fn forward(
        &self,
        compute: &Compute,
        input: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        let expected = rows.checked_mul(self.input).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "{LABEL}: linear rows * input overflows: {rows} * {}",
                self.input
            ))
        })?;
        if input.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: linear input has {} values, expected {rows} x {} = {expected}",
                input.len(),
                self.input
            )));
        }
        let output_len = rows.checked_mul(self.output).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "{LABEL}: linear rows * output overflows: {rows} * {}",
                self.output
            ))
        })?;
        let mut output = vec![0.0f32; output_len];
        compute.gemm_f32(
            rows,
            self.output,
            self.input,
            input,
            &self.weight_t,
            self.bias.as_deref(),
            &mut output,
        )?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub(super) struct QuantizerWeights {
    pub(super) codebooks: Vec<Vec<f32>>,
    pub(super) projections: Vec<Linear>,
    pub(super) output_projection: Linear,
}

#[derive(Debug, Clone)]
pub(super) struct LayerWeights {
    pub(super) norm1_weight: Vec<f32>,
    pub(super) norm1_bias: Vec<f32>,
    pub(super) attention_in: Linear,
    pub(super) attention_out: Linear,
    pub(super) layer_scale1: Vec<f32>,
    pub(super) norm2_weight: Vec<f32>,
    pub(super) norm2_bias: Vec<f32>,
    pub(super) ffn_in: Linear,
    pub(super) ffn_out: Linear,
    pub(super) layer_scale2: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct StageWeights {
    pub(super) spec: StageSpec,
    pub(super) input_projection: Linear,
    pub(super) layers: Vec<LayerWeights>,
    pub(super) output_projection: Linear,
}

#[derive(Debug, Clone)]
pub(super) struct NanoWeights {
    pub(super) quantizer: QuantizerWeights,
    pub(super) stages: Vec<StageWeights>,
}

impl NanoWeights {
    pub(super) fn bind(file: &GgufFile) -> Result<Self> {
        let mut codebooks = Vec::with_capacity(NUM_QUANTIZERS);
        let mut projections = Vec::with_capacity(NUM_QUANTIZERS);
        for index in 0..NUM_QUANTIZERS {
            let prefix = format!("quantizer.quantizers.{index}");
            codebooks.push(load_tensor(
                file,
                LABEL,
                &format!("{prefix}.codebook.weight"),
                &[CODEBOOK_SIZE, CODEBOOK_DIM],
            )?);
            projections.push(Linear::load_weight_norm_1x1(
                file,
                &format!("{prefix}.out_proj"),
                CODEBOOK_DIM,
                RVQ_DIM,
            )?);
        }
        let quantizer = QuantizerWeights {
            codebooks,
            projections,
            output_projection: Linear::load_weight_norm_1x1(
                file,
                "quantizer.output_proj",
                RVQ_DIM,
                LATENT_DIM,
            )?,
        };

        let stages = STAGE_SPECS
            .iter()
            .copied()
            .map(|spec| StageWeights::bind(file, spec))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { quantizer, stages })
    }
}

impl StageWeights {
    fn bind(file: &GgufFile, spec: StageSpec) -> Result<Self> {
        let prefix = format!("decoder.{}", spec.module_index);
        let input_projection = Linear::load(
            file,
            &format!("{prefix}.input_proj.weight"),
            spec.input_dim,
            MODEL_DIM,
            None,
        )?;
        let mut layers = Vec::with_capacity(spec.layers);
        for index in 0..spec.layers {
            layers.push(LayerWeights::bind(
                file,
                &format!("{prefix}.transformer.layers.{index}"),
            )?);
        }
        let output_projection = Linear::load(
            file,
            &format!("{prefix}.output_proj.weight"),
            MODEL_DIM,
            spec.output_dim,
            None,
        )?;
        Ok(Self {
            spec,
            input_projection,
            layers,
            output_projection,
        })
    }
}

impl LayerWeights {
    fn bind(file: &GgufFile, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm1_weight: load_tensor(
                file,
                LABEL,
                &format!("{prefix}.norm1.weight"),
                &[MODEL_DIM],
            )?,
            norm1_bias: load_tensor(file, LABEL, &format!("{prefix}.norm1.bias"), &[MODEL_DIM])?,
            attention_in: Linear::load(
                file,
                &format!("{prefix}.self_attn.in_proj.weight"),
                MODEL_DIM,
                MODEL_DIM * 3,
                None,
            )?,
            attention_out: Linear::load(
                file,
                &format!("{prefix}.self_attn.out_proj.weight"),
                MODEL_DIM,
                MODEL_DIM,
                None,
            )?,
            layer_scale1: load_tensor(
                file,
                LABEL,
                &format!("{prefix}.layer_scale_1.scale"),
                &[MODEL_DIM],
            )?,
            norm2_weight: load_tensor(
                file,
                LABEL,
                &format!("{prefix}.norm2.weight"),
                &[MODEL_DIM],
            )?,
            norm2_bias: load_tensor(file, LABEL, &format!("{prefix}.norm2.bias"), &[MODEL_DIM])?,
            ffn_in: Linear::load(
                file,
                &format!("{prefix}.ffn.0.weight"),
                MODEL_DIM,
                FFN_DIM,
                None,
            )?,
            ffn_out: Linear::load(
                file,
                &format!("{prefix}.ffn.2.weight"),
                FFN_DIM,
                MODEL_DIM,
                None,
            )?,
            layer_scale2: load_tensor(
                file,
                LABEL,
                &format!("{prefix}.layer_scale_2.scale"),
                &[MODEL_DIM],
            )?,
        })
    }
}

fn transpose_pytorch_linear(weight: &[f32], input: usize, output: usize) -> Vec<f32> {
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
    input: usize,
    output: usize,
    label: &str,
) -> Result<Vec<f32>> {
    if magnitude.len() != output || direction.len() != output * input {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: weight norm `{label}` shape mismatch: magnitude={}, direction={}, expected {output} and {}",
            magnitude.len(),
            direction.len(),
            output * input
        )));
    }
    let mut weight = vec![0.0f32; direction.len()];
    for out in 0..output {
        let row = &direction[out * input..(out + 1) * input];
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 || !magnitude[out].is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: weight norm `{label}` row {out} has invalid magnitude {} or direction norm {norm}",
                magnitude[out]
            )));
        }
        let scale = magnitude[out] / norm;
        for inner in 0..input {
            weight[out * input + inner] = row[inner] * scale;
        }
    }
    Ok(weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pytorch_linear_is_transposed_for_compute() {
        assert_eq!(
            transpose_pytorch_linear(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn weight_norm_is_folded_per_output_row() {
        let weight = fold_weight_norm_rows(&[5.0, 13.0], &[3.0, 4.0, 5.0, 12.0], 2, 2, "x")
            .expect("valid rows");
        assert_eq!(weight, vec![3.0, 4.0, 5.0, 12.0]);
    }
}
