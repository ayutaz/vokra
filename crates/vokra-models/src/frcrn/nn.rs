//! Native FRCRN forward lowered to the shared `Compute` seam.

use vokra_core::{Result, VokraError};

use crate::compute::Compute;

use super::weights::{
    AffineWeights, BatchNormWeights, ComplexBatchNormWeights, ComplexConvWeights,
    ComplexFsmnL1Weights, ComplexFsmnWeights, FrcrnWeights, RealFsmnWeights, SePathWeights,
    SeWeights, UnetWeights,
};
use super::{CHANNELS, FEATURE_DIM, FFT_LENGTH, FSMN_ORDER, HOP_LENGTH, LABEL};

const BN_EPS: f32 = 1.0e-5;
const LEAKY_RELU_SLOPE: f32 = 0.01;

#[derive(Debug, Clone)]
struct ComplexTensor {
    re: Vec<f32>,
    im: Vec<f32>,
    channels: usize,
    height: usize,
    width: usize,
}

impl ComplexTensor {
    fn zeros(channels: usize, height: usize, width: usize) -> Result<Self> {
        let len = checked_product("complex tensor", &[channels, height, width])?;
        Ok(Self {
            re: vec![0.0; len],
            im: vec![0.0; len],
            channels,
            height,
            width,
        })
    }

    fn validate(&self, label: &str) -> Result<()> {
        let expected = checked_product(label, &[self.channels, self.height, self.width])?;
        if self.re.len() != expected || self.im.len() != expected {
            return Err(invalid(format!(
                "{label} payloads are re={} im={}, expected {expected} for [{},{},{}]",
                self.re.len(),
                self.im.len(),
                self.channels,
                self.height,
                self.width
            )));
        }
        Ok(())
    }

    fn index(&self, channel: usize, height: usize, width: usize) -> usize {
        (channel * self.height + height) * self.width + width
    }

    fn cat_channels(&self, other: &Self) -> Result<Self> {
        self.validate("concat left")?;
        other.validate("concat right")?;
        if self.height != other.height || self.width != other.width {
            return Err(invalid(format!(
                "skip concat spatial mismatch: left=[{},{},{}], right=[{},{},{}]",
                self.channels, self.height, self.width, other.channels, other.height, other.width
            )));
        }
        let mut output = Self::zeros(self.channels + other.channels, self.height, self.width)?;
        output.re[..self.re.len()].copy_from_slice(&self.re);
        output.im[..self.im.len()].copy_from_slice(&self.im);
        output.re[self.re.len()..].copy_from_slice(&other.re);
        output.im[self.im.len()..].copy_from_slice(&other.im);
        Ok(output)
    }
}

pub(super) fn enhance(compute: &Compute, weights: &FrcrnWeights, pcm: &[f32]) -> Result<Vec<f32>> {
    validate_pcm(pcm)?;
    let spectrum = stft(weights, pcm)?;
    let unet1 = unet_forward(compute, &weights.unet, &spectrum)?;
    let mask1 = complex_tanh(&unet1);
    let unet2 = unet_forward(compute, &weights.unet2, &unet1)?;
    let mut mask = complex_tanh(&unet2);
    if mask.re.len() != mask1.re.len() {
        return Err(invalid("the two U-Net mask shapes diverged"));
    }
    for index in 0..mask.re.len() {
        mask.re[index] += mask1.re[index];
        mask.im[index] += mask1.im[index];
    }
    let enhanced = complex_multiply(&spectrum, &mask)?;
    let waveform = istft(weights, &enhanced)?;
    if let Some((index, _)) = waveform
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(invalid(format!(
            "enhanced waveform contains a non-finite sample at {index}"
        )));
    }
    Ok(waveform)
}

fn validate_pcm(pcm: &[f32]) -> Result<()> {
    if pcm.len() < FFT_LENGTH {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: input has {} samples, but the fixed STFT requires at least {FFT_LENGTH}",
            pcm.len()
        )));
    }
    if let Some((index, _)) = pcm.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: input PCM sample {index} is not finite"
        )));
    }
    Ok(())
}

fn stft(weights: &FrcrnWeights, pcm: &[f32]) -> Result<ComplexTensor> {
    let frames = (pcm.len() - FFT_LENGTH) / HOP_LENGTH + 1;
    let mut output = ComplexTensor::zeros(1, FEATURE_DIM, frames)?;
    for frame in 0..frames {
        let input_offset = frame * HOP_LENGTH;
        for bin in 0..FEATURE_DIM {
            let mut real = 0.0;
            let mut imag = 0.0;
            let real_weight = bin * FFT_LENGTH;
            let imag_weight = (FEATURE_DIM + bin) * FFT_LENGTH;
            for sample in 0..FFT_LENGTH {
                let value = pcm[input_offset + sample];
                real += value * weights.stft[real_weight + sample];
                imag += value * weights.stft[imag_weight + sample];
            }
            let index = bin * frames + frame;
            output.re[index] = real;
            output.im[index] = imag;
        }
    }
    Ok(output)
}

fn istft(weights: &FrcrnWeights, spectrum: &ComplexTensor) -> Result<Vec<f32>> {
    spectrum.validate("iSTFT spectrum")?;
    if spectrum.channels != 1 || spectrum.height != FEATURE_DIM || spectrum.width == 0 {
        return Err(invalid(format!(
            "iSTFT expected [1,{FEATURE_DIM},frames], got [{},{},{}]",
            spectrum.channels, spectrum.height, spectrum.width
        )));
    }
    let frames = spectrum.width;
    let output_len = (frames - 1)
        .checked_mul(HOP_LENGTH)
        .and_then(|value| value.checked_add(FFT_LENGTH))
        .ok_or_else(|| invalid("iSTFT output length overflows"))?;
    let mut output = vec![0.0; output_len];
    for frame in 0..frames {
        let output_offset = frame * HOP_LENGTH;
        for bin in 0..FEATURE_DIM {
            let index = bin * frames + frame;
            let real = spectrum.re[index];
            let imag = spectrum.im[index];
            let real_weight = bin * FFT_LENGTH;
            let imag_weight = (FEATURE_DIM + bin) * FFT_LENGTH;
            for sample in 0..FFT_LENGTH {
                output[output_offset + sample] += real * weights.istft[real_weight + sample]
                    + imag * weights.istft[imag_weight + sample];
            }
        }
    }
    let mut coefficient = vec![0.0; output_len];
    for frame in 0..frames {
        let offset = frame * HOP_LENGTH;
        for sample in 0..FFT_LENGTH {
            let window = weights.window[sample];
            coefficient[offset + sample] += window * window;
        }
    }
    for (value, coefficient) in output.iter_mut().zip(coefficient) {
        *value /= coefficient + 1.0e-8;
    }
    Ok(output)
}

fn unet_forward(
    compute: &Compute,
    weights: &UnetWeights,
    input: &ComplexTensor,
) -> Result<ComplexTensor> {
    if weights.encoders.len() != 7
        || weights.decoders.len() != 7
        || weights.fsmn_enc.len() != 6
        || weights.fsmn_dec.len() != 6
        || weights.se_enc.len() != 7
        || weights.se_dec.len() != 5
    {
        return Err(invalid("bound U-Net vector lengths do not match depth 14"));
    }
    let mut x = input.clone();
    let mut skip = Vec::with_capacity(8);
    skip.push(input.clone());
    for layer in 0..7 {
        if layer > 0 {
            x = complex_fsmn_l1(compute, &weights.fsmn_enc[layer - 1], &x)?;
        }
        let kernel_h = if layer == 6 { 2 } else { 5 };
        x = complex_conv2d(
            compute,
            &x,
            &weights.encoders[layer].conv,
            CHANNELS,
            kernel_h,
            2,
            2,
            1,
            0,
            1,
        )?;
        apply_complex_bn(&mut x, &weights.encoders[layer].bn)?;
        leaky_relu(&mut x);
        skip.push(se_forward(compute, &weights.se_enc[layer], &x)?);
    }

    let mut p = complex_fsmn(compute, &weights.fsmn, &x)?;
    let decoder_kernels = [2, 5, 5, 5, 6, 5, 5];
    let decoder_channels = [128, 128, 128, 128, 128, 128, 1];
    for layer in 0..7 {
        p = complex_conv_transpose2d(
            compute,
            &p,
            &weights.decoders[layer].conv,
            decoder_channels[layer],
            decoder_kernels[layer],
            2,
            2,
            1,
            0,
            1,
        )?;
        apply_complex_bn(&mut p, &weights.decoders[layer].bn)?;
        leaky_relu(&mut p);
        if layer < 6 {
            p = complex_fsmn_l1(compute, &weights.fsmn_dec[layer], &p)?;
        }
        if layer == 6 {
            break;
        }
        if layer < 5 {
            p = se_forward(compute, &weights.se_dec[layer], &p)?;
        }
        p = p.cat_channels(&skip[6 - layer])?;
    }
    complex_conv2d(compute, &p, &weights.linear, 1, 1, 1, 1, 1, 0, 0)
}

#[allow(clippy::too_many_arguments)]
fn complex_conv2d(
    compute: &Compute,
    input: &ComplexTensor,
    weights: &ComplexConvWeights,
    output_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
) -> Result<ComplexTensor> {
    input.validate("Conv2d input")?;
    if stride_h == 0
        || stride_w == 0
        || input.height + 2 * padding_h < kernel_h
        || input.width + 2 * padding_w < kernel_w
    {
        return Err(invalid("invalid Conv2d geometry"));
    }
    let output_h = (input.height + 2 * padding_h - kernel_h) / stride_h + 1;
    let output_w = (input.width + 2 * padding_w - kernel_w) / stride_w + 1;
    let positions = checked_product("Conv2d positions", &[output_h, output_w])?;
    let kernel_size = checked_product("Conv2d kernel", &[input.channels, kernel_h, kernel_w])?;
    require_len(
        "Conv2d real weight",
        weights.re_gemm.len(),
        output_channels * kernel_size,
    )?;
    require_len(
        "Conv2d imaginary weight",
        weights.im_gemm.len(),
        output_channels * kernel_size,
    )?;
    require_len("Conv2d real bias", weights.re_bias.len(), output_channels)?;
    require_len(
        "Conv2d imaginary bias",
        weights.im_bias.len(),
        output_channels,
    )?;

    let mut columns_re = vec![0.0; positions * kernel_size];
    let mut columns_im = vec![0.0; positions * kernel_size];
    for oh in 0..output_h {
        for ow in 0..output_w {
            let row = oh * output_w + ow;
            for channel in 0..input.channels {
                for kh in 0..kernel_h {
                    let padded_h = oh * stride_h + kh;
                    let Some(ih) = padded_h.checked_sub(padding_h) else {
                        continue;
                    };
                    if ih >= input.height {
                        continue;
                    }
                    for kw in 0..kernel_w {
                        let padded_w = ow * stride_w + kw;
                        let Some(iw) = padded_w.checked_sub(padding_w) else {
                            continue;
                        };
                        if iw >= input.width {
                            continue;
                        }
                        let column = (channel * kernel_h + kh) * kernel_w + kw;
                        let source = input.index(channel, ih, iw);
                        columns_re[row * kernel_size + column] = input.re[source];
                        columns_im[row * kernel_size + column] = input.im[source];
                    }
                }
            }
        }
    }
    let mut re_re = vec![0.0; positions * output_channels];
    let mut im_im = vec![0.0; positions * output_channels];
    let mut re_im = vec![0.0; positions * output_channels];
    let mut im_re = vec![0.0; positions * output_channels];
    compute.gemm_f32(
        positions,
        output_channels,
        kernel_size,
        &columns_re,
        &weights.re_gemm,
        None,
        &mut re_re,
    )?;
    compute.gemm_f32(
        positions,
        output_channels,
        kernel_size,
        &columns_im,
        &weights.im_gemm,
        None,
        &mut im_im,
    )?;
    compute.gemm_f32(
        positions,
        output_channels,
        kernel_size,
        &columns_im,
        &weights.re_gemm,
        None,
        &mut re_im,
    )?;
    compute.gemm_f32(
        positions,
        output_channels,
        kernel_size,
        &columns_re,
        &weights.im_gemm,
        None,
        &mut im_re,
    )?;

    let mut output = ComplexTensor::zeros(output_channels, output_h, output_w)?;
    for row in 0..positions {
        let oh = row / output_w;
        let ow = row % output_w;
        for channel in 0..output_channels {
            let source = row * output_channels + channel;
            let destination = output.index(channel, oh, ow);
            output.re[destination] =
                re_re[source] - im_im[source] + weights.re_bias[channel] - weights.im_bias[channel];
            output.im[destination] =
                re_im[source] + im_re[source] + weights.re_bias[channel] + weights.im_bias[channel];
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn complex_conv_transpose2d(
    compute: &Compute,
    input: &ComplexTensor,
    weights: &ComplexConvWeights,
    output_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
) -> Result<ComplexTensor> {
    input.validate("ConvTranspose2d input")?;
    if input.height == 0 || input.width == 0 || stride_h == 0 || stride_w == 0 {
        return Err(invalid("invalid ConvTranspose2d geometry"));
    }
    let output_h = transpose_extent(input.height, stride_h, padding_h, kernel_h)?;
    let output_w = transpose_extent(input.width, stride_w, padding_w, kernel_w)?;
    let positions = checked_product("ConvTranspose2d positions", &[input.height, input.width])?;
    let expanded = checked_product(
        "ConvTranspose2d expanded width",
        &[output_channels, kernel_h, kernel_w],
    )?;
    let weight_len = checked_product("ConvTranspose2d weight", &[input.channels, expanded])?;
    require_len(
        "ConvTranspose2d real weight",
        weights.re_gemm.len(),
        weight_len,
    )?;
    require_len(
        "ConvTranspose2d imaginary weight",
        weights.im_gemm.len(),
        weight_len,
    )?;
    require_len(
        "ConvTranspose2d real bias",
        weights.re_bias.len(),
        output_channels,
    )?;
    require_len(
        "ConvTranspose2d imaginary bias",
        weights.im_bias.len(),
        output_channels,
    )?;

    let mut rows_re = vec![0.0; positions * input.channels];
    let mut rows_im = vec![0.0; positions * input.channels];
    for ih in 0..input.height {
        for iw in 0..input.width {
            let row = ih * input.width + iw;
            for channel in 0..input.channels {
                let source = input.index(channel, ih, iw);
                rows_re[row * input.channels + channel] = input.re[source];
                rows_im[row * input.channels + channel] = input.im[source];
            }
        }
    }
    let mut re_re = vec![0.0; positions * expanded];
    let mut im_im = vec![0.0; positions * expanded];
    let mut re_im = vec![0.0; positions * expanded];
    let mut im_re = vec![0.0; positions * expanded];
    compute.gemm_f32(
        positions,
        expanded,
        input.channels,
        &rows_re,
        &weights.re_gemm,
        None,
        &mut re_re,
    )?;
    compute.gemm_f32(
        positions,
        expanded,
        input.channels,
        &rows_im,
        &weights.im_gemm,
        None,
        &mut im_im,
    )?;
    compute.gemm_f32(
        positions,
        expanded,
        input.channels,
        &rows_im,
        &weights.re_gemm,
        None,
        &mut re_im,
    )?;
    compute.gemm_f32(
        positions,
        expanded,
        input.channels,
        &rows_re,
        &weights.im_gemm,
        None,
        &mut im_re,
    )?;

    let mut output = ComplexTensor::zeros(output_channels, output_h, output_w)?;
    for ih in 0..input.height {
        for iw in 0..input.width {
            let row = ih * input.width + iw;
            for channel in 0..output_channels {
                for kh in 0..kernel_h {
                    let Some(oh) = (ih * stride_h + kh).checked_sub(padding_h) else {
                        continue;
                    };
                    if oh >= output_h {
                        continue;
                    }
                    for kw in 0..kernel_w {
                        let Some(ow) = (iw * stride_w + kw).checked_sub(padding_w) else {
                            continue;
                        };
                        if ow >= output_w {
                            continue;
                        }
                        let expanded_index = (channel * kernel_h + kh) * kernel_w + kw;
                        let source = row * expanded + expanded_index;
                        let destination = output.index(channel, oh, ow);
                        output.re[destination] += re_re[source] - im_im[source];
                        output.im[destination] += re_im[source] + im_re[source];
                    }
                }
            }
        }
    }
    let plane = output_h * output_w;
    for channel in 0..output_channels {
        let re_bias = weights.re_bias[channel] - weights.im_bias[channel];
        let im_bias = weights.re_bias[channel] + weights.im_bias[channel];
        for index in channel * plane..(channel + 1) * plane {
            output.re[index] += re_bias;
            output.im[index] += im_bias;
        }
    }
    Ok(output)
}

fn transpose_extent(input: usize, stride: usize, padding: usize, kernel: usize) -> Result<usize> {
    (input - 1)
        .checked_mul(stride)
        .and_then(|value| value.checked_add(kernel))
        .and_then(|value| value.checked_sub(2 * padding))
        .ok_or_else(|| invalid("ConvTranspose2d output extent overflows/underflows"))
}

fn apply_complex_bn(tensor: &mut ComplexTensor, weights: &ComplexBatchNormWeights) -> Result<()> {
    apply_bn(
        &mut tensor.re,
        tensor.channels,
        tensor.height,
        tensor.width,
        &weights.re,
    )?;
    apply_bn(
        &mut tensor.im,
        tensor.channels,
        tensor.height,
        tensor.width,
        &weights.im,
    )
}

fn apply_bn(
    values: &mut [f32],
    channels: usize,
    height: usize,
    width: usize,
    weights: &BatchNormWeights,
) -> Result<()> {
    let plane = checked_product("BatchNorm plane", &[height, width])?;
    require_len("BatchNorm activation", values.len(), channels * plane)?;
    for (name, len) in [
        ("weight", weights.weight.len()),
        ("bias", weights.bias.len()),
        ("running_mean", weights.running_mean.len()),
        ("running_var", weights.running_var.len()),
    ] {
        require_len(&format!("BatchNorm {name}"), len, channels)?;
    }
    for channel in 0..channels {
        let scale = weights.weight[channel] / (weights.running_var[channel] + BN_EPS).sqrt();
        let shift = weights.bias[channel] - weights.running_mean[channel] * scale;
        for value in &mut values[channel * plane..(channel + 1) * plane] {
            *value = *value * scale + shift;
        }
    }
    Ok(())
}

fn leaky_relu(tensor: &mut ComplexTensor) {
    for value in tensor.re.iter_mut().chain(tensor.im.iter_mut()) {
        if *value < 0.0 {
            *value *= LEAKY_RELU_SLOPE;
        }
    }
}

fn se_forward(
    compute: &Compute,
    weights: &SeWeights,
    input: &ComplexTensor,
) -> Result<ComplexTensor> {
    input.validate("SE input")?;
    if input.channels != CHANNELS {
        return Err(invalid(format!(
            "SE expected {CHANNELS} channels, got {}",
            input.channels
        )));
    }
    let plane = input.height * input.width;
    let mut pooled_re = vec![0.0; CHANNELS];
    let mut pooled_im = vec![0.0; CHANNELS];
    for channel in 0..CHANNELS {
        let range = channel * plane..(channel + 1) * plane;
        pooled_re[channel] = input.re[range.clone()].iter().sum::<f32>() / plane as f32;
        pooled_im[channel] = input.im[range].iter().sum::<f32>() / plane as f32;
    }
    let rr = se_path(compute, &weights.re, &pooled_re)?;
    let ri = se_path(compute, &weights.re, &pooled_im)?;
    let ii = se_path(compute, &weights.im, &pooled_im)?;
    let ir = se_path(compute, &weights.im, &pooled_re)?;
    let mut output = input.clone();
    for channel in 0..CHANNELS {
        let scale_re = rr[channel] - ii[channel];
        let scale_im = ri[channel] + ir[channel];
        for index in channel * plane..(channel + 1) * plane {
            output.re[index] *= scale_re;
            output.im[index] *= scale_im;
        }
    }
    Ok(output)
}

fn se_path(compute: &Compute, weights: &SePathWeights, input: &[f32]) -> Result<Vec<f32>> {
    let mut hidden = affine(compute, &weights.down, input)?;
    for value in &mut hidden {
        *value = value.max(0.0);
    }
    let mut output = affine(compute, &weights.up, &hidden)?;
    for value in &mut output {
        *value = sigmoid(*value);
    }
    Ok(output)
}

fn complex_fsmn_l1(
    compute: &Compute,
    weights: &ComplexFsmnL1Weights,
    input: &ComplexTensor,
) -> Result<ComplexTensor> {
    input.validate("frequency FSMN input")?;
    if input.channels != CHANNELS {
        return Err(invalid(format!(
            "frequency FSMN expected {CHANNELS} features, got {}",
            input.channels
        )));
    }
    let batches = input.width;
    let sequence = input.height;
    let mut rows_re = vec![0.0; batches * sequence * CHANNELS];
    let mut rows_im = vec![0.0; rows_re.len()];
    for batch in 0..batches {
        for step in 0..sequence {
            for feature in 0..CHANNELS {
                let row = (batch * sequence + step) * CHANNELS + feature;
                let source = input.index(feature, step, batch);
                rows_re[row] = input.re[source];
                rows_im[row] = input.im[source];
            }
        }
    }
    let (result_re, result_im) = complex_fsmn_pair(
        compute,
        &weights.re,
        &weights.im,
        &rows_re,
        &rows_im,
        batches,
        sequence,
    )?;
    let mut output = ComplexTensor::zeros(input.channels, input.height, input.width)?;
    for batch in 0..batches {
        for step in 0..sequence {
            for feature in 0..CHANNELS {
                let row = (batch * sequence + step) * CHANNELS + feature;
                let destination = output.index(feature, step, batch);
                output.re[destination] = result_re[row];
                output.im[destination] = result_im[row];
            }
        }
    }
    Ok(output)
}

fn complex_fsmn(
    compute: &Compute,
    weights: &ComplexFsmnWeights,
    input: &ComplexTensor,
) -> Result<ComplexTensor> {
    input.validate("central FSMN input")?;
    let features = input.channels * input.height;
    if features != CHANNELS {
        return Err(invalid(format!(
            "central FSMN expected flattened feature width {CHANNELS}, got {features}"
        )));
    }
    let sequence = input.width;
    let mut rows_re = vec![0.0; sequence * CHANNELS];
    let mut rows_im = vec![0.0; rows_re.len()];
    for step in 0..sequence {
        for channel in 0..input.channels {
            for height in 0..input.height {
                let feature = channel * input.height + height;
                let source = input.index(channel, height, step);
                rows_re[step * CHANNELS + feature] = input.re[source];
                rows_im[step * CHANNELS + feature] = input.im[source];
            }
        }
    }
    let (l1_re, l1_im) = complex_fsmn_pair(
        compute,
        &weights.re_l1,
        &weights.im_l1,
        &rows_re,
        &rows_im,
        1,
        sequence,
    )?;
    let (result_re, result_im) = complex_fsmn_pair(
        compute,
        &weights.re_l2,
        &weights.im_l2,
        &l1_re,
        &l1_im,
        1,
        sequence,
    )?;
    let mut output = ComplexTensor::zeros(input.channels, input.height, input.width)?;
    for step in 0..sequence {
        for channel in 0..input.channels {
            for height in 0..input.height {
                let feature = channel * input.height + height;
                let destination = output.index(channel, height, step);
                output.re[destination] = result_re[step * CHANNELS + feature];
                output.im[destination] = result_im[step * CHANNELS + feature];
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn complex_fsmn_pair(
    compute: &Compute,
    re_weights: &RealFsmnWeights,
    im_weights: &RealFsmnWeights,
    input_re: &[f32],
    input_im: &[f32],
    batches: usize,
    sequence: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let re_re = real_fsmn(compute, re_weights, input_re, batches, sequence)?;
    let im_im = real_fsmn(compute, im_weights, input_im, batches, sequence)?;
    let re_im = real_fsmn(compute, re_weights, input_im, batches, sequence)?;
    let im_re = real_fsmn(compute, im_weights, input_re, batches, sequence)?;
    let mut output_re = vec![0.0; input_re.len()];
    let mut output_im = vec![0.0; input_re.len()];
    for index in 0..input_re.len() {
        output_re[index] = re_re[index] - im_im[index];
        output_im[index] = re_im[index] + im_re[index];
    }
    Ok((output_re, output_im))
}

fn real_fsmn(
    compute: &Compute,
    weights: &RealFsmnWeights,
    input: &[f32],
    batches: usize,
    sequence: usize,
) -> Result<Vec<f32>> {
    let rows = checked_product("FSMN rows", &[batches, sequence])?;
    require_len("FSMN input", input.len(), rows * CHANNELS)?;
    let mut hidden = affine(compute, &weights.linear, input)?;
    for value in &mut hidden {
        *value = value.max(0.0);
    }
    let projected = affine(compute, &weights.project, &hidden)?;

    let padded_len = sequence
        .checked_add(FSMN_ORDER - 1)
        .ok_or_else(|| invalid("FSMN padded length overflows"))?;
    let grouped_channels = checked_product("FSMN grouped channels", &[batches, CHANNELS])?;
    let mut padded = vec![0.0; grouped_channels * padded_len];
    for batch in 0..batches {
        for feature in 0..CHANNELS {
            let destination = (batch * CHANNELS + feature) * padded_len + FSMN_ORDER - 1;
            for step in 0..sequence {
                padded[destination + step] =
                    projected[(batch * sequence + step) * CHANNELS + feature];
            }
        }
    }
    require_len(
        "FSMN depthwise kernel",
        weights.conv.len(),
        CHANNELS * FSMN_ORDER,
    )?;
    let mut repeated_weights = vec![0.0; grouped_channels * FSMN_ORDER];
    for batch in 0..batches {
        repeated_weights[batch * CHANNELS * FSMN_ORDER..(batch + 1) * CHANNELS * FSMN_ORDER]
            .copy_from_slice(&weights.conv);
    }
    let mut convolved = vec![0.0; grouped_channels * sequence];
    compute.grouped_conv1d_f32(
        &padded,
        grouped_channels,
        padded_len,
        &repeated_weights,
        grouped_channels,
        FSMN_ORDER,
        None,
        1,
        0,
        grouped_channels,
        &mut convolved,
    )?;
    let mut output = vec![0.0; input.len()];
    for batch in 0..batches {
        for step in 0..sequence {
            for feature in 0..CHANNELS {
                let row = (batch * sequence + step) * CHANNELS + feature;
                let conv = (batch * CHANNELS + feature) * sequence + step;
                output[row] = input[row] + projected[row] + convolved[conv];
            }
        }
    }
    Ok(output)
}

fn affine(compute: &Compute, weights: &AffineWeights, input: &[f32]) -> Result<Vec<f32>> {
    if weights.in_features == 0
        || weights.out_features == 0
        || input.is_empty()
        || input.len() % weights.in_features != 0
    {
        return Err(invalid(format!(
            "affine input {} is incompatible with {} -> {}",
            input.len(),
            weights.in_features,
            weights.out_features
        )));
    }
    require_len(
        "affine transposed weight",
        weights.weight_t.len(),
        weights.in_features * weights.out_features,
    )?;
    if let Some(bias) = &weights.bias {
        require_len("affine bias", bias.len(), weights.out_features)?;
    }
    let rows = input.len() / weights.in_features;
    let mut output = vec![0.0; rows * weights.out_features];
    compute.gemm_f32(
        rows,
        weights.out_features,
        weights.in_features,
        input,
        &weights.weight_t,
        weights.bias.as_deref(),
        &mut output,
    )?;
    Ok(output)
}

fn complex_tanh(input: &ComplexTensor) -> ComplexTensor {
    let mut output = input.clone();
    for value in output.re.iter_mut().chain(output.im.iter_mut()) {
        *value = value.tanh();
    }
    output
}

fn complex_multiply(left: &ComplexTensor, right: &ComplexTensor) -> Result<ComplexTensor> {
    left.validate("complex multiply left")?;
    right.validate("complex multiply right")?;
    if (left.channels, left.height, left.width) != (right.channels, right.height, right.width) {
        return Err(invalid("complex multiply shape mismatch"));
    }
    let mut output = ComplexTensor::zeros(left.channels, left.height, left.width)?;
    for index in 0..left.re.len() {
        output.re[index] = left.re[index] * right.re[index] - left.im[index] * right.im[index];
        output.im[index] = left.re[index] * right.im[index] + left.im[index] * right.re[index];
    }
    Ok(output)
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn checked_product(label: &str, dimensions: &[usize]) -> Result<usize> {
    dimensions.iter().try_fold(1usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or_else(|| invalid(format!("{label} shape overflows: {dimensions:?}")))
    })
}

fn require_len(label: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(invalid(format!(
            "{label} has {actual} elements, expected {expected}"
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> VokraError {
    VokraError::InvalidArgument(format!("{LABEL}: {}", message.into()))
}
