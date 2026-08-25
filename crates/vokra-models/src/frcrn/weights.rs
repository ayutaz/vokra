//! Strict FRCRN tensor decoding.

use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::strict_checkpoint::load_tensor;

use super::{CHANNELS, FEATURE_DIM, FFT_LENGTH, FSMN_ORDER, LABEL, SE_HIDDEN};

#[derive(Debug, Clone)]
pub(super) struct AffineWeights {
    pub(super) weight_t: Vec<f32>,
    pub(super) bias: Option<Vec<f32>>,
    pub(super) in_features: usize,
    pub(super) out_features: usize,
}

#[derive(Debug, Clone)]
pub(super) struct ComplexConvWeights {
    /// GEMM-ready matrix: `[in*kh*kw,out]` for Conv2d and
    /// `[in,out*kh*kw]` for ConvTranspose2d.
    pub(super) re_gemm: Vec<f32>,
    pub(super) re_bias: Vec<f32>,
    pub(super) im_gemm: Vec<f32>,
    pub(super) im_bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct BatchNormWeights {
    pub(super) weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) running_mean: Vec<f32>,
    pub(super) running_var: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct ComplexBatchNormWeights {
    pub(super) re: BatchNormWeights,
    pub(super) im: BatchNormWeights,
}

#[derive(Debug, Clone)]
pub(super) struct ConvBlockWeights {
    pub(super) conv: ComplexConvWeights,
    pub(super) bn: ComplexBatchNormWeights,
}

#[derive(Debug, Clone)]
pub(super) struct RealFsmnWeights {
    pub(super) linear: AffineWeights,
    pub(super) project: AffineWeights,
    pub(super) conv: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct ComplexFsmnL1Weights {
    pub(super) re: RealFsmnWeights,
    pub(super) im: RealFsmnWeights,
}

#[derive(Debug, Clone)]
pub(super) struct ComplexFsmnWeights {
    pub(super) re_l1: RealFsmnWeights,
    pub(super) im_l1: RealFsmnWeights,
    pub(super) re_l2: RealFsmnWeights,
    pub(super) im_l2: RealFsmnWeights,
}

#[derive(Debug, Clone)]
pub(super) struct SePathWeights {
    pub(super) down: AffineWeights,
    pub(super) up: AffineWeights,
}

#[derive(Debug, Clone)]
pub(super) struct SeWeights {
    pub(super) re: SePathWeights,
    pub(super) im: SePathWeights,
}

#[derive(Debug, Clone)]
pub(super) struct UnetWeights {
    pub(super) encoders: Vec<ConvBlockWeights>,
    pub(super) decoders: Vec<ConvBlockWeights>,
    /// Official forward uses indices 1..=6; checkpoint index 0 is dead state.
    pub(super) fsmn_enc: Vec<ComplexFsmnL1Weights>,
    /// Official forward uses indices 0..=5; checkpoint index 6 is dead state.
    pub(super) fsmn_dec: Vec<ComplexFsmnL1Weights>,
    pub(super) fsmn: ComplexFsmnWeights,
    pub(super) se_enc: Vec<SeWeights>,
    /// Official forward uses indices 0..=4; checkpoint index 5 is dead state.
    pub(super) se_dec: Vec<SeWeights>,
    pub(super) linear: ComplexConvWeights,
}

#[derive(Debug, Clone)]
pub(super) struct FrcrnWeights {
    pub(super) stft: Vec<f32>,
    pub(super) istft: Vec<f32>,
    pub(super) window: Vec<f32>,
    pub(super) unet: UnetWeights,
    pub(super) unet2: UnetWeights,
}

impl FrcrnWeights {
    pub(super) fn bind(file: &GgufFile) -> Result<Self> {
        let stft = tensor(file, "stft.weight", &[2 * FEATURE_DIM, 1, FFT_LENGTH])?;
        let istft = tensor(file, "istft.weight", &[2 * FEATURE_DIM, 1, FFT_LENGTH])?;
        let window = tensor(file, "istft.window", &[1, FFT_LENGTH, 1])?;
        let enframe = tensor(file, "istft.enframe", &[FFT_LENGTH, 1, FFT_LENGTH])?;
        validate_enframe(&enframe)?;
        Ok(Self {
            stft,
            istft,
            window,
            unet: load_unet(file, "unet")?,
            unet2: load_unet(file, "unet2")?,
        })
    }
}

fn load_unet(file: &GgufFile, root: &str) -> Result<UnetWeights> {
    let mut encoders = Vec::with_capacity(7);
    for layer in 0..7 {
        let input_channels = if layer == 0 { 1 } else { CHANNELS };
        let kernel_h = if layer == 6 { 2 } else { 5 };
        let prefix = format!("{root}.encoder{layer}");
        encoders.push(ConvBlockWeights {
            conv: load_complex_conv(
                file,
                &format!("{prefix}.conv"),
                "conv",
                &[CHANNELS, input_channels, kernel_h, 2],
                CHANNELS,
            )?,
            bn: load_complex_bn(file, &format!("{prefix}.bn"), CHANNELS)?,
        });
    }

    let decoder_geometry = [
        (128, 128, 2),
        (256, 128, 5),
        (256, 128, 5),
        (256, 128, 5),
        (256, 128, 6),
        (256, 128, 5),
        (256, 1, 5),
    ];
    let mut decoders = Vec::with_capacity(7);
    for (layer, &(input_channels, output_channels, kernel_h)) in decoder_geometry.iter().enumerate()
    {
        let prefix = format!("{root}.decoder{layer}");
        decoders.push(ConvBlockWeights {
            conv: load_complex_conv(
                file,
                &format!("{prefix}.transconv"),
                "tconv",
                &[input_channels, output_channels, kernel_h, 2],
                output_channels,
            )?,
            bn: load_complex_bn(file, &format!("{prefix}.bn"), output_channels)?,
        });
    }

    let mut fsmn_enc = Vec::with_capacity(6);
    for layer in 1..7 {
        fsmn_enc.push(load_complex_fsmn_l1(
            file,
            &format!("{root}.fsmn_enc{layer}"),
        )?);
    }
    let mut fsmn_dec = Vec::with_capacity(6);
    for layer in 0..6 {
        fsmn_dec.push(load_complex_fsmn_l1(
            file,
            &format!("{root}.fsmn_dec{layer}"),
        )?);
    }
    let mut se_enc = Vec::with_capacity(7);
    for layer in 0..7 {
        se_enc.push(load_se(file, &format!("{root}.se_layer_enc{layer}"))?);
    }
    let mut se_dec = Vec::with_capacity(5);
    for layer in 0..5 {
        se_dec.push(load_se(file, &format!("{root}.se_layer_dec{layer}"))?);
    }
    Ok(UnetWeights {
        encoders,
        decoders,
        fsmn_enc,
        fsmn_dec,
        fsmn: load_complex_fsmn(file, &format!("{root}.fsmn"))?,
        se_enc,
        se_dec,
        linear: load_complex_conv(file, &format!("{root}.linear"), "conv", &[1, 1, 1, 1], 1)?,
    })
}

fn load_complex_conv(
    file: &GgufFile,
    prefix: &str,
    stem: &str,
    shape: &[usize],
    bias: usize,
) -> Result<ComplexConvWeights> {
    let re_weight = tensor(file, &format!("{prefix}.{stem}_re.weight"), shape)?;
    let im_weight = tensor(file, &format!("{prefix}.{stem}_im.weight"), shape)?;
    let (re_gemm, im_gemm) = match stem {
        "conv" => (
            conv2d_gemm_matrix(&re_weight, shape)?,
            conv2d_gemm_matrix(&im_weight, shape)?,
        ),
        "tconv" => (re_weight, im_weight),
        _ => {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: unsupported complex-convolution tensor stem {stem:?}"
            )));
        }
    };
    Ok(ComplexConvWeights {
        re_gemm,
        re_bias: tensor(file, &format!("{prefix}.{stem}_re.bias"), &[bias])?,
        im_gemm,
        im_bias: tensor(file, &format!("{prefix}.{stem}_im.bias"), &[bias])?,
    })
}

fn conv2d_gemm_matrix(weight: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    if shape.len() != 4 {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: Conv2d weight shape must be rank four, got {shape:?}"
        )));
    }
    let output = shape[0];
    let input = shape[1]
        .checked_mul(shape[2])
        .and_then(|value| value.checked_mul(shape[3]))
        .ok_or_else(|| VokraError::ModelLoad(format!("{LABEL}: Conv2d shape overflows")))?;
    let mut transposed = vec![0.0; weight.len()];
    for out in 0..output {
        for inner in 0..input {
            transposed[inner * output + out] = weight[out * input + inner];
        }
    }
    Ok(transposed)
}

fn load_complex_bn(
    file: &GgufFile,
    prefix: &str,
    channels: usize,
) -> Result<ComplexBatchNormWeights> {
    Ok(ComplexBatchNormWeights {
        re: load_bn(file, &format!("{prefix}.bn_re"), channels)?,
        im: load_bn(file, &format!("{prefix}.bn_im"), channels)?,
    })
}

fn load_bn(file: &GgufFile, prefix: &str, channels: usize) -> Result<BatchNormWeights> {
    Ok(BatchNormWeights {
        weight: tensor(file, &format!("{prefix}.weight"), &[channels])?,
        bias: tensor(file, &format!("{prefix}.bias"), &[channels])?,
        running_mean: tensor(file, &format!("{prefix}.running_mean"), &[channels])?,
        running_var: tensor(file, &format!("{prefix}.running_var"), &[channels])?,
    })
}

fn load_complex_fsmn_l1(file: &GgufFile, prefix: &str) -> Result<ComplexFsmnL1Weights> {
    Ok(ComplexFsmnL1Weights {
        re: load_real_fsmn(file, &format!("{prefix}.fsmn_re_L1"))?,
        im: load_real_fsmn(file, &format!("{prefix}.fsmn_im_L1"))?,
    })
}

fn load_complex_fsmn(file: &GgufFile, prefix: &str) -> Result<ComplexFsmnWeights> {
    Ok(ComplexFsmnWeights {
        re_l1: load_real_fsmn(file, &format!("{prefix}.fsmn_re_L1"))?,
        im_l1: load_real_fsmn(file, &format!("{prefix}.fsmn_im_L1"))?,
        re_l2: load_real_fsmn(file, &format!("{prefix}.fsmn_re_L2"))?,
        im_l2: load_real_fsmn(file, &format!("{prefix}.fsmn_im_L2"))?,
    })
}

fn load_real_fsmn(file: &GgufFile, prefix: &str) -> Result<RealFsmnWeights> {
    Ok(RealFsmnWeights {
        linear: load_affine(file, &format!("{prefix}.linear"), CHANNELS, CHANNELS, true)?,
        project: load_affine(
            file,
            &format!("{prefix}.project"),
            CHANNELS,
            CHANNELS,
            false,
        )?,
        conv: tensor(
            file,
            &format!("{prefix}.conv1.weight"),
            &[CHANNELS, 1, FSMN_ORDER, 1],
        )?,
    })
}

fn load_se(file: &GgufFile, prefix: &str) -> Result<SeWeights> {
    Ok(SeWeights {
        re: load_se_path(file, &format!("{prefix}.fc_r"))?,
        im: load_se_path(file, &format!("{prefix}.fc_i"))?,
    })
}

fn load_se_path(file: &GgufFile, prefix: &str) -> Result<SePathWeights> {
    Ok(SePathWeights {
        down: load_affine(file, &format!("{prefix}.0"), CHANNELS, SE_HIDDEN, true)?,
        up: load_affine(file, &format!("{prefix}.2"), SE_HIDDEN, CHANNELS, true)?,
    })
}

fn load_affine(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    has_bias: bool,
) -> Result<AffineWeights> {
    let weight = tensor(file, &format!("{prefix}.weight"), &[output, input])?;
    let mut weight_t = vec![0.0; weight.len()];
    for out in 0..output {
        for inner in 0..input {
            weight_t[inner * output + out] = weight[out * input + inner];
        }
    }
    Ok(AffineWeights {
        weight_t,
        bias: has_bias
            .then(|| tensor(file, &format!("{prefix}.bias"), &[output]))
            .transpose()?,
        in_features: input,
        out_features: output,
    })
}

fn tensor(file: &GgufFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let values = load_tensor(file, LABEL, name, shape)?;
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{name}` contains a non-finite value at flat index {index}"
        )));
    }
    Ok(values)
}

fn validate_enframe(values: &[f32]) -> Result<()> {
    for row in 0..FFT_LENGTH {
        for column in 0..FFT_LENGTH {
            let expected = if row == column { 1.0 } else { 0.0 };
            let value = values[row * FFT_LENGTH + column];
            if value != expected {
                return Err(VokraError::ModelLoad(format!(
                    "{LABEL}: `istft.enframe` is not the fixed {FFT_LENGTH}x{FFT_LENGTH} identity at [{row},{column}]: {value}"
                )));
            }
        }
    }
    Ok(())
}
