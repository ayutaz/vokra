use vokra_core::Result;
use vokra_core::gguf::{GgmlType, GgufFile};

use crate::strict_checkpoint::load_tensor;

use super::{DEPTH, HIDDEN, KERNEL_SIZE, LSTM_HIDDEN};

const LABEL: &str = "facebook_denoiser";

#[derive(Debug, Clone)]
pub(super) struct Conv1d {
    pub(super) weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) kernel: usize,
}

#[derive(Debug, Clone)]
pub(super) struct ConvTranspose1d {
    /// PyTorch ConvTranspose1d layout `[input, output, kernel]`. Flattening
    /// the final two axes gives the row-major `[input, output * kernel]`
    /// matrix consumed by the backend GEMM lowering.
    pub(super) weight_flat: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) kernel: usize,
}

#[derive(Debug, Clone)]
pub(super) struct EncoderStage {
    pub(super) downsample: Conv1d,
    pub(super) project: Conv1d,
}

#[derive(Debug, Clone)]
pub(super) struct DecoderStage {
    pub(super) project: Conv1d,
    pub(super) upsample: ConvTranspose1d,
}

#[derive(Debug, Clone)]
pub(super) struct LstmLayer {
    /// PyTorch `[4H, H]`, transposed to backend GEMM `[H, 4H]`.
    pub(super) weight_ih_t: Vec<f32>,
    /// PyTorch `[4H, H]`, retained for backend GEMV per recurrent step.
    pub(super) weight_hh: Vec<f32>,
    pub(super) bias_ih: Vec<f32>,
    pub(super) bias_hh: Vec<f32>,
}

#[derive(Debug, Clone)]
/// Fully decoded F32 weights for the immutable 48-tensor DNS48 release.
pub struct FbDenoiserWeights {
    pub(super) encoder: Vec<EncoderStage>,
    pub(super) lstm: Vec<LstmLayer>,
    pub(super) decoder: Vec<DecoderStage>,
}

impl FbDenoiserWeights {
    /// Strictly binds all 48 tensors of the audited DNS48 checkpoint.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        crate::strict_checkpoint::StrictCheckpoint::bind(file, super::SPEC)?;
        Self::bind(file)
    }

    /// Number of tensors in the immutable DNS48 manifest.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        super::TENSOR_COUNT
    }

    pub(super) fn bind(file: &GgufFile) -> Result<Self> {
        let channels = [48usize, 96, 192, 384, 768];
        debug_assert_eq!(channels[0], HIDDEN);
        debug_assert_eq!(channels.len(), DEPTH);

        let mut encoder = Vec::with_capacity(DEPTH);
        let mut input = 1usize;
        for (stage, &output) in channels.iter().enumerate() {
            encoder.push(EncoderStage {
                downsample: load_conv(
                    file,
                    &format!("encoder.{stage}.0"),
                    input,
                    output,
                    KERNEL_SIZE,
                )?,
                project: load_conv(file, &format!("encoder.{stage}.2"), output, 2 * output, 1)?,
            });
            input = output;
        }

        let mut lstm = Vec::with_capacity(2);
        for layer in 0..2 {
            let weight_ih = load_finite(
                file,
                &format!("lstm.lstm.weight_ih_l{layer}"),
                &[4 * LSTM_HIDDEN, LSTM_HIDDEN],
            )?;
            lstm.push(LstmLayer {
                weight_ih_t: transpose_out_in(&weight_ih, 4 * LSTM_HIDDEN, LSTM_HIDDEN),
                weight_hh: load_finite(
                    file,
                    &format!("lstm.lstm.weight_hh_l{layer}"),
                    &[4 * LSTM_HIDDEN, LSTM_HIDDEN],
                )?,
                bias_ih: load_finite(
                    file,
                    &format!("lstm.lstm.bias_ih_l{layer}"),
                    &[4 * LSTM_HIDDEN],
                )?,
                bias_hh: load_finite(
                    file,
                    &format!("lstm.lstm.bias_hh_l{layer}"),
                    &[4 * LSTM_HIDDEN],
                )?,
            });
        }

        let decoder_inputs = [768usize, 384, 192, 96, 48];
        let decoder_outputs = [384usize, 192, 96, 48, 1];
        let mut decoder = Vec::with_capacity(DEPTH);
        for (stage, (&input, &output)) in decoder_inputs
            .iter()
            .zip(decoder_outputs.iter())
            .enumerate()
        {
            decoder.push(DecoderStage {
                project: load_conv(file, &format!("decoder.{stage}.0"), input, 2 * input, 1)?,
                upsample: load_conv_transpose(
                    file,
                    &format!("decoder.{stage}.2"),
                    input,
                    output,
                    KERNEL_SIZE,
                )?,
            });
        }
        Ok(Self {
            encoder,
            lstm,
            decoder,
        })
    }
}

fn load_conv(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
) -> Result<Conv1d> {
    Ok(Conv1d {
        weight: load_finite(file, &format!("{prefix}.weight"), &[output, input, kernel])?,
        bias: load_finite(file, &format!("{prefix}.bias"), &[output])?,
        input,
        output,
        kernel,
    })
}

fn load_conv_transpose(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
) -> Result<ConvTranspose1d> {
    Ok(ConvTranspose1d {
        weight_flat: load_finite(file, &format!("{prefix}.weight"), &[input, output, kernel])?,
        bias: load_finite(file, &format!("{prefix}.bias"), &[output])?,
        input,
        output,
        kernel,
    })
}

fn transpose_out_in(weight: &[f32], output: usize, input: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; input * output];
    for out in 0..output {
        for inner in 0..input {
            transposed[inner * output + out] = weight[out * input + inner];
        }
    }
    transposed
}

fn load_finite(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let dtype = file
        .tensor_info(name)
        .ok_or_else(|| {
            vokra_core::VokraError::ModelLoad(format!(
                "{LABEL}: required tensor `{name}` is missing"
            ))
        })?
        .dtype;
    if dtype != GgmlType::F32 {
        return Err(vokra_core::VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{name}` is {dtype:?}, expected exact public F32 DNS48 weights"
        )));
    }
    let values = load_tensor(file, LABEL, name, expected)?;
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(vokra_core::VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{name}` contains a non-finite value at element {index}"
        )));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pytorch_linear_transpose_is_row_major() {
        let source = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            transpose_out_in(&source, 2, 3),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }
}
