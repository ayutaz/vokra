//! Backend seam for complete BERT-family transformer forwards.
//!
//! `vokra-bert` deliberately does not depend on a concrete device backend.
//! Model runtimes provide one implementation of this trait for the selected
//! backend, while embedding lookup, residual addition and relative-position
//! index gathering remain host-side control/layout work.

use vokra_core::Result;

/// Learned primitives required by the BERT, DeBERTa v2 and DeBERTa v3
/// encoders.
///
/// A caller must provide every method from one selected backend. Implementors
/// must return an error for an unsupported primitive; they must never execute
/// that primitive on a different backend as a fallback (FR-EX-08).
pub trait BertBackendOps {
    /// Row-major linear projection.
    ///
    /// `input` is `[rows, input_dim]`, `weight_out_in` is
    /// `[output_dim, input_dim]`, and `output` is `[rows, output_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn linear_f32(
        &self,
        input: &[f32],
        weight_out_in: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        output: &mut [f32],
    ) -> Result<()>;

    /// Row-wise softmax over `[rows, cols]`.
    fn softmax_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()>;

    /// Affine row-wise LayerNorm over `[rows, cols]`.
    #[allow(clippy::too_many_arguments)]
    fn layer_norm_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()>;

    /// Element-wise exact GELU.
    fn gelu_f32(&self, input: &[f32], output: &mut [f32]) -> Result<()>;

    /// Channel-major Conv1D.
    ///
    /// `input` is `[input_channels, input_len]`, `weight` is
    /// `[output_channels, input_channels, kernel]`, and `output` is
    /// `[output_channels, output_len]`.
    #[allow(clippy::too_many_arguments)]
    fn conv1d_f32(
        &self,
        input: &[f32],
        input_channels: usize,
        input_len: usize,
        weight: &[f32],
        output_channels: usize,
        kernel: usize,
        bias: Option<&[f32]>,
        stride: usize,
        padding: usize,
        output: &mut [f32],
    ) -> Result<()>;
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_with_backend(
    backend: &dyn BertBackendOps,
    input: &[f32],
    weight_out_in: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0; rows * output_dim];
    backend.linear_f32(
        input,
        weight_out_in,
        bias,
        rows,
        input_dim,
        output_dim,
        &mut output,
    )?;
    Ok(output)
}

pub(crate) fn gather_head(
    input: &[f32],
    rows: usize,
    row_width: usize,
    head_offset: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; rows * head_dim];
    for row in 0..rows {
        output[row * head_dim..(row + 1) * head_dim].copy_from_slice(
            &input[row * row_width + head_offset..row * row_width + head_offset + head_dim],
        );
    }
    output
}

pub(crate) fn transpose_rows(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = input[row * cols + col];
        }
    }
    output
}

#[cfg(test)]
pub(crate) struct TestBackend;

#[cfg(test)]
impl BertBackendOps for TestBackend {
    fn linear_f32(
        &self,
        input: &[f32],
        weight_out_in: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        output: &mut [f32],
    ) -> Result<()> {
        assert_eq!(input.len(), rows * input_dim);
        assert_eq!(weight_out_in.len(), output_dim * input_dim);
        assert_eq!(output.len(), rows * output_dim);
        if let Some(bias) = bias {
            assert_eq!(bias.len(), output_dim);
        }
        for row in 0..rows {
            for out_channel in 0..output_dim {
                let mut sum = bias.map(|values| values[out_channel]).unwrap_or(0.0);
                for input_channel in 0..input_dim {
                    sum += input[row * input_dim + input_channel]
                        * weight_out_in[out_channel * input_dim + input_channel];
                }
                output[row * output_dim + out_channel] = sum;
            }
        }
        Ok(())
    }

    fn softmax_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        assert_eq!(input.len(), rows * cols);
        assert_eq!(output.len(), input.len());
        for row in 0..rows {
            let start = row * cols;
            let maximum = input[start..start + cols]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for col in 0..cols {
                let value = vokra_math::exp(input[start + col] - maximum);
                output[start + col] = value;
                sum += value;
            }
            for col in 0..cols {
                output[start + col] /= sum;
            }
        }
        Ok(())
    }

    fn layer_norm_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        assert_eq!(input.len(), rows * cols);
        assert_eq!(output.len(), input.len());
        assert_eq!(gamma.len(), cols);
        assert_eq!(beta.len(), cols);
        for row in 0..rows {
            let start = row * cols;
            let mean = input[start..start + cols].iter().sum::<f32>() / cols as f32;
            let variance = input[start..start + cols]
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f32>()
                / cols as f32;
            let inverse = 1.0 / vokra_math::sqrt(variance + eps);
            for col in 0..cols {
                output[start + col] =
                    (input[start + col] - mean) * inverse * gamma[col] + beta[col];
            }
        }
        Ok(())
    }

    fn gelu_f32(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        assert_eq!(output.len(), input.len());
        for (output, input) in output.iter_mut().zip(input) {
            *output = crate::bert_base::gelu_exact(*input);
        }
        Ok(())
    }

    fn conv1d_f32(
        &self,
        input: &[f32],
        input_channels: usize,
        input_len: usize,
        weight: &[f32],
        output_channels: usize,
        kernel: usize,
        bias: Option<&[f32]>,
        stride: usize,
        padding: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let output_len = (input_len + 2 * padding - kernel) / stride + 1;
        assert_eq!(input.len(), input_channels * input_len);
        assert_eq!(weight.len(), output_channels * input_channels * kernel);
        assert_eq!(output.len(), output_channels * output_len);
        for out_channel in 0..output_channels {
            for out_position in 0..output_len {
                let mut sum = bias.map(|values| values[out_channel]).unwrap_or(0.0);
                for input_channel in 0..input_channels {
                    for kernel_position in 0..kernel {
                        let input_position = out_position * stride + kernel_position;
                        if input_position < padding || input_position - padding >= input_len {
                            continue;
                        }
                        sum += input[input_channel * input_len + input_position - padding]
                            * weight[(out_channel * input_channels + input_channel) * kernel
                                + kernel_position];
                    }
                }
                output[out_channel * output_len + out_position] = sum;
            }
        }
        Ok(())
    }
}
