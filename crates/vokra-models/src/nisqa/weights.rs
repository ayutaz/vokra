use vokra_core::gguf::{GgmlType, GgufFile};
use vokra_core::{Result, VokraError};

use crate::strict_checkpoint::load_tensor;

use super::{SPEC, TENSOR_COUNT};

const LABEL: &str = "nisqa";

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
pub(super) struct BatchNorm {
    pub(super) gamma: Vec<f32>,
    pub(super) beta: Vec<f32>,
    pub(super) running_mean: Vec<f32>,
    pub(super) running_var: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct ConvBlock {
    pub(super) conv: Conv2d,
    pub(super) norm: BatchNorm,
}

#[derive(Debug, Clone)]
pub(super) struct Linear {
    /// Row-major `[input, output]`, transposed once from PyTorch `[output, input]`.
    pub(super) weight_io: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) input: usize,
    pub(super) output: usize,
}

#[derive(Debug, Clone)]
pub(super) struct Norm {
    pub(super) gamma: Vec<f32>,
    pub(super) beta: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct AttentionLayer {
    pub(super) in_proj: Linear,
    pub(super) out_proj: Linear,
    pub(super) linear1: Linear,
    pub(super) linear2: Linear,
    pub(super) norm1: Norm,
    pub(super) norm2: Norm,
}

#[derive(Debug, Clone)]
pub(super) struct PoolHead {
    pub(super) linear1: Linear,
    pub(super) linear2: Linear,
    pub(super) linear3: Linear,
}

/// Fully decoded immutable public NISQA multidimensional checkpoint.
#[derive(Debug, Clone)]
pub struct NisqaWeights {
    pub(super) conv: Vec<ConvBlock>,
    pub(super) input_linear: Linear,
    pub(super) input_norm: Norm,
    pub(super) attention: Vec<AttentionLayer>,
    pub(super) pool_heads: Vec<PoolHead>,
}

impl NisqaWeights {
    /// Strictly binds the complete official 94-tensor manifest.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        crate::strict_checkpoint::StrictCheckpoint::bind(file, SPEC)?;
        Self::bind(file)
    }

    pub(super) fn bind(file: &GgufFile) -> Result<Self> {
        let channels = [1usize, 16, 32, 64, 64, 64, 64];
        let mut conv = Vec::with_capacity(6);
        for layer in 1..=6 {
            let input = channels[layer - 1];
            let output = channels[layer];
            conv.push(ConvBlock {
                conv: load_conv(file, &format!("cnn.model.conv{layer}"), input, output)?,
                norm: load_batch_norm(file, &format!("cnn.model.bn{layer}"), output)?,
            });
        }

        let input_linear = load_linear(file, "time_dependency.model.linear", 384, 64)?;
        let input_norm = load_norm(file, "time_dependency.model.norm1", 64)?;
        let mut attention = Vec::with_capacity(2);
        for layer in 0..2 {
            let prefix = format!("time_dependency.model.layers.{layer}");
            attention.push(AttentionLayer {
                in_proj: load_packed_attention(file, &format!("{prefix}.self_attn.in_proj"))?,
                out_proj: load_linear(file, &format!("{prefix}.self_attn.out_proj"), 64, 64)?,
                linear1: load_linear(file, &format!("{prefix}.linear1"), 64, 64)?,
                linear2: load_linear(file, &format!("{prefix}.linear2"), 64, 64)?,
                norm1: load_norm(file, &format!("{prefix}.norm1"), 64)?,
                norm2: load_norm(file, &format!("{prefix}.norm2"), 64)?,
            });
        }

        let mut pool_heads = Vec::with_capacity(5);
        for head in 0..5 {
            let prefix = format!("pool_layers.{head}.model");
            pool_heads.push(PoolHead {
                linear1: load_linear(file, &format!("{prefix}.linear1"), 64, 128)?,
                linear2: load_linear(file, &format!("{prefix}.linear2"), 128, 1)?,
                linear3: load_linear(file, &format!("{prefix}.linear3"), 64, 1)?,
            });
        }

        Ok(Self {
            conv,
            input_linear,
            input_norm,
            attention,
            pool_heads,
        })
    }

    /// Immutable learned-tensor count.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        TENSOR_COUNT
    }
}

fn load_conv(file: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<Conv2d> {
    Ok(Conv2d {
        weight: load_finite(file, &format!("{prefix}.weight"), &[output, input, 3, 3])?,
        bias: load_finite(file, &format!("{prefix}.bias"), &[output])?,
        input,
        output,
        kernel_h: 3,
        kernel_w: 3,
    })
}

fn load_batch_norm(file: &GgufFile, prefix: &str, channels: usize) -> Result<BatchNorm> {
    Ok(BatchNorm {
        gamma: load_finite(file, &format!("{prefix}.weight"), &[channels])?,
        beta: load_finite(file, &format!("{prefix}.bias"), &[channels])?,
        running_mean: load_finite(file, &format!("{prefix}.running_mean"), &[channels])?,
        running_var: load_finite(file, &format!("{prefix}.running_var"), &[channels])?,
    })
}

fn load_linear(file: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<Linear> {
    load_linear_tensors(
        file,
        &format!("{prefix}.weight"),
        &format!("{prefix}.bias"),
        input,
        output,
    )
}

fn load_packed_attention(file: &GgufFile, prefix: &str) -> Result<Linear> {
    load_linear_tensors(
        file,
        &format!("{prefix}_weight"),
        &format!("{prefix}_bias"),
        64,
        192,
    )
}

fn load_linear_tensors(
    file: &GgufFile,
    weight_name: &str,
    bias_name: &str,
    input: usize,
    output: usize,
) -> Result<Linear> {
    let weight = load_finite(file, weight_name, &[output, input])?;
    Ok(Linear {
        weight_io: transpose_out_in(&weight, output, input),
        bias: load_finite(file, bias_name, &[output])?,
        input,
        output,
    })
}

fn load_norm(file: &GgufFile, prefix: &str, width: usize) -> Result<Norm> {
    Ok(Norm {
        gamma: load_finite(file, &format!("{prefix}.weight"), &[width])?,
        beta: load_finite(file, &format!("{prefix}.bias"), &[width])?,
    })
}

fn transpose_out_in(values: &[f32], output: usize, input: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; input * output];
    for out in 0..output {
        for inner in 0..input {
            transposed[inner * output + out] = values[out * input + inner];
        }
    }
    transposed
}

fn load_finite(file: &GgufFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("{LABEL}: required tensor `{name}` is missing"))
    })?;
    if info.dtype != GgmlType::F32 {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{name}` is {:?}, expected exact public F32 weights",
            info.dtype
        )));
    }
    let values = load_tensor(file, LABEL, name, shape)?;
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{name}` contains a non-finite value at index {index}"
        )));
    }
    Ok(values)
}
