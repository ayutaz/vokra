use vokra_core::gguf::{GgmlType, GgufFile};
use vokra_core::{Result, VokraError};

use crate::strict_checkpoint::load_tensor;

use super::{P808_CNN_CHANNELS, P835_BINS, P835_WINDOW, TENSOR_COUNT};

const LABEL: &str = "dnsmos";

#[derive(Debug, Clone)]
pub(super) struct Conv2d {
    pub(super) weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) input: usize,
    pub(super) output: usize,
}

#[derive(Debug, Clone)]
pub(super) struct Dense {
    /// ONNX MatMul layout `[input, output]`, consumed directly by GEMM.
    pub(super) weight_io: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) input: usize,
    pub(super) output: usize,
}

#[derive(Debug, Clone)]
pub(super) struct P808Weights {
    pub(super) conv: Vec<Conv2d>,
    pub(super) dense: Vec<Dense>,
}

#[derive(Debug, Clone)]
pub(super) struct P835Weights {
    /// ONNX Conv1d layout `[bins, window, 1]`, flattened to `[window, bins]`.
    pub(super) stft_real_io: Vec<f32>,
    pub(super) stft_imag_io: Vec<f32>,
    pub(super) conv: Vec<Conv2d>,
    pub(super) dense: Vec<Dense>,
}

#[derive(Debug, Clone)]
/// Fully decoded weights for the immutable public 38-tensor DNSMOS bundle.
pub struct DnsmosWeights {
    pub(super) p808: P808Weights,
    pub(super) p835: P835Weights,
}

impl DnsmosWeights {
    /// Strictly binds the complete official DNSMOS tensor manifest.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        crate::strict_checkpoint::StrictCheckpoint::bind(file, super::SPEC)?;
        Self::bind(file)
    }

    pub(super) fn bind(file: &GgufFile) -> Result<Self> {
        let p808_channels = [1usize, 32, 32, 32, 32, 64];
        let mut p808_conv = Vec::with_capacity(5);
        for layer in 0..5 {
            p808_conv.push(load_conv(
                file,
                &format!("p808.conv2d_{}", layer + 5),
                p808_channels[layer],
                p808_channels[layer + 1],
            )?);
        }
        let p808_dense = vec![
            load_dense(
                file,
                "p808.mos_estimator_small_1/dense_3",
                P808_CNN_CHANNELS,
                64,
            )?,
            load_dense(file, "p808.mos_estimator_small_1/dense_4", 64, 64)?,
            load_dense(file, "p808.mos_estimator_small_1/dense_5", 64, 1)?,
        ];

        let stft_real = load_finite(
            file,
            "p835.time2freq/stft-real/kernel:0",
            &[P835_BINS, P835_WINDOW, 1],
        )?;
        let stft_imag = load_finite(
            file,
            "p835.time2freq/stft-imag/kernel:0",
            &[P835_BINS, P835_WINDOW, 1],
        )?;
        let p835_channels = [1usize, 128, 64, 64, 32, 32, 32, 64];
        let mut p835_conv = Vec::with_capacity(7);
        for layer in 0..7 {
            let suffix = if layer == 0 {
                String::new()
            } else {
                format!("_{layer}")
            };
            p835_conv.push(load_conv(
                file,
                &format!("p835.conv2d{suffix}"),
                p835_channels[layer],
                p835_channels[layer + 1],
            )?);
        }
        let p835_dense = vec![
            load_dense(file, "p835.mos_estimator_logpow/dense", 64, 128)?,
            load_dense(file, "p835.mos_estimator_logpow/dense_1", 128, 64)?,
            load_dense(file, "p835.mos_estimator_logpow/dense_3", 64, 3)?,
        ];

        Ok(Self {
            p808: P808Weights {
                conv: p808_conv,
                dense: p808_dense,
            },
            p835: P835Weights {
                stft_real_io: transpose_out_in(&stft_real, P835_BINS, P835_WINDOW),
                stft_imag_io: transpose_out_in(&stft_imag, P835_BINS, P835_WINDOW),
                conv: p835_conv,
                dense: p835_dense,
            },
        })
    }

    /// Returns the authenticated checkpoint tensor count.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        TENSOR_COUNT
    }

    pub(super) fn synthesized() -> Self {
        let conv = |input: usize, output: usize| Conv2d {
            weight: vec![0.0; output * input * 3 * 3],
            bias: vec![0.0; output],
            input,
            output,
        };
        let dense = |input: usize, output: usize| Dense {
            weight_io: vec![0.0; input * output],
            bias: vec![0.0; output],
            input,
            output,
        };
        Self {
            p808: P808Weights {
                conv: vec![
                    conv(1, 32),
                    conv(32, 32),
                    conv(32, 32),
                    conv(32, 32),
                    conv(32, 64),
                ],
                dense: vec![dense(P808_CNN_CHANNELS, 64), dense(64, 64), dense(64, 1)],
            },
            p835: P835Weights {
                stft_real_io: vec![0.0; P835_WINDOW * P835_BINS],
                stft_imag_io: vec![0.0; P835_WINDOW * P835_BINS],
                conv: vec![
                    conv(1, 128),
                    conv(128, 64),
                    conv(64, 64),
                    conv(64, 32),
                    conv(32, 32),
                    conv(32, 32),
                    conv(32, 64),
                ],
                dense: vec![dense(64, 128), dense(128, 64), dense(64, 3)],
            },
        }
    }
}

fn load_conv(file: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<Conv2d> {
    Ok(Conv2d {
        weight: load_finite(file, &format!("{prefix}/kernel:0"), &[output, input, 3, 3])?,
        bias: load_finite(file, &format!("{prefix}/bias:0"), &[output])?,
        input,
        output,
    })
}

fn load_dense(file: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<Dense> {
    Ok(Dense {
        weight_io: load_finite(
            file,
            &format!("{prefix}/MatMul/ReadVariableOp/resource:0"),
            &[input, output],
        )?,
        bias: load_finite(
            file,
            &format!("{prefix}/BiasAdd/ReadVariableOp/resource:0"),
            &[output],
        )?,
        input,
        output,
    })
}

fn transpose_out_in(values: &[f32], output: usize, input: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; input * output];
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
            "{LABEL}: tensor `{name}` contains a non-finite value at element {index}"
        )));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onnx_conv1d_kernel_transposes_to_gemm_io() {
        assert_eq!(
            transpose_out_in(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }
}
