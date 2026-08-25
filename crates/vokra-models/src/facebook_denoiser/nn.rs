use std::f32::consts::PI;

use vokra_core::{Result, VokraError};

use crate::compute::Compute;

use super::weights::{Conv1d, ConvTranspose1d, FbDenoiserWeights, LstmLayer};
use super::{DEPTH, KERNEL_SIZE, LSTM_HIDDEN, NORMALIZATION_FLOOR, RESAMPLE_ZEROS, STRIDE};

pub(super) fn denoise(
    compute: &Compute,
    weights: &FbDenoiserWeights,
    pcm: &[f32],
) -> Result<Vec<f32>> {
    let original_length = pcm.len();
    let (mut x, standard_deviation) = normalize(pcm)?;
    x.resize(valid_length(original_length), 0.0);
    x = upsample2(&x, 1);
    x = upsample2(&x, 1);

    let mut channels = 1usize;
    let mut length = x.len();
    let mut skips = Vec::with_capacity(DEPTH);
    for stage in &weights.encoder {
        let (encoded, encoded_length) = conv1d(compute, &x, length, &stage.downsample, STRIDE)?;
        x = encoded;
        relu_in_place(&mut x);
        let (projected, projected_length) = conv1d(compute, &x, encoded_length, &stage.project, 1)?;
        debug_assert_eq!(projected_length, encoded_length);
        x = glu(&projected, stage.downsample.output, encoded_length)?;
        channels = stage.downsample.output;
        length = encoded_length;
        skips.push(x.clone());
    }
    if channels != LSTM_HIDDEN {
        return Err(shape_error(format!(
            "encoder produced {channels} channels, expected {LSTM_HIDDEN}"
        )));
    }

    let mut sequence = channel_to_time_major(&x, channels, length)?;
    for layer in &weights.lstm {
        sequence = lstm_layer(compute, &sequence, length, layer)?;
    }
    x = time_to_channel_major(&sequence, length, channels)?;

    for (stage_index, stage) in weights.decoder.iter().enumerate() {
        let skip = skips
            .pop()
            .ok_or_else(|| shape_error("decoder is missing its paired encoder skip"))?;
        add_cropped_skip(&mut x, &skip, stage.project.input, length)?;
        let (projected, projected_length) = conv1d(compute, &x, length, &stage.project, 1)?;
        debug_assert_eq!(projected_length, length);
        x = glu(&projected, stage.project.input, length)?;
        let (decoded, decoded_length) =
            conv_transpose1d(compute, &x, length, &stage.upsample, STRIDE)?;
        x = decoded;
        length = decoded_length;
        channels = stage.upsample.output;
        if stage_index + 1 < weights.decoder.len() {
            relu_in_place(&mut x);
        }
    }
    debug_assert!(skips.is_empty());
    if channels != 1 {
        return Err(shape_error(format!(
            "decoder produced {channels} channels, expected mono output"
        )));
    }

    x = downsample2(&x, 1);
    x = downsample2(&x, 1);
    if x.len() < original_length {
        return Err(shape_error(format!(
            "decoder produced {} samples, shorter than requested {original_length}",
            x.len()
        )));
    }
    x.truncate(original_length);
    for value in &mut x {
        *value *= standard_deviation;
    }
    if let Some((index, _)) = x.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "facebook_denoiser: output sample {index} is not finite"
        )));
    }
    Ok(x)
}

fn normalize(pcm: &[f32]) -> Result<(Vec<f32>, f32)> {
    if pcm.len() < 2 {
        return Err(VokraError::InvalidArgument(
            "facebook_denoiser: input PCM needs at least two samples for the official correction=1 standard deviation"
                .to_owned(),
        ));
    }
    if let Some((index, _)) = pcm.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "facebook_denoiser: input PCM sample {index} is not finite"
        )));
    }
    let mean = pcm.iter().sum::<f32>() / pcm.len() as f32;
    let variance = pcm
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f32>()
        / (pcm.len() - 1) as f32;
    let standard_deviation = variance.sqrt();
    if !standard_deviation.is_finite() {
        return Err(VokraError::InvalidArgument(
            "facebook_denoiser: input standard deviation is not finite".to_owned(),
        ));
    }
    let divisor = NORMALIZATION_FLOOR + standard_deviation;
    Ok((
        pcm.iter().map(|value| value / divisor).collect(),
        standard_deviation,
    ))
}

pub(super) fn valid_length(samples: usize) -> usize {
    let mut length = i64::try_from(samples)
        .unwrap_or(i64::MAX / 4)
        .saturating_mul(4);
    for _ in 0..DEPTH {
        length = ceil_div(length - KERNEL_SIZE as i64, STRIDE as i64) + 1;
        length = length.max(1);
    }
    for _ in 0..DEPTH {
        length = (length - 1) * STRIDE as i64 + KERNEL_SIZE as i64;
    }
    usize::try_from(ceil_div(length, 4)).unwrap_or(usize::MAX)
}

fn ceil_div(value: i64, divisor: i64) -> i64 {
    (value + divisor - 1).div_euclid(divisor)
}

fn conv1d(
    compute: &Compute,
    input: &[f32],
    input_length: usize,
    conv: &Conv1d,
    stride: usize,
) -> Result<(Vec<f32>, usize)> {
    if input.len() != conv.input * input_length || input_length < conv.kernel {
        return Err(shape_error(format!(
            "Conv1d input {} is not [{} channels, {input_length} samples] or is shorter than kernel {}",
            input.len(),
            conv.input,
            conv.kernel
        )));
    }
    let output_length = (input_length - conv.kernel) / stride + 1;
    let mut output = vec![0.0f32; conv.output * output_length];
    compute.conv1d_f32(
        input,
        conv.input,
        input_length,
        &conv.weight,
        conv.output,
        conv.kernel,
        Some(&conv.bias),
        stride,
        0,
        &mut output,
    )?;
    Ok((output, output_length))
}

fn glu(input: &[f32], output_channels: usize, length: usize) -> Result<Vec<f32>> {
    if input.len() != 2 * output_channels * length {
        return Err(shape_error(format!(
            "GLU input {} is not [2 * {output_channels}, {length}]",
            input.len()
        )));
    }
    let split = output_channels * length;
    let mut output = vec![0.0f32; split];
    for index in 0..split {
        output[index] = input[index] * sigmoid(input[split + index]);
    }
    Ok(output)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponent = value.exp();
        exponent / (1.0 + exponent)
    }
}

fn lstm_layer(
    compute: &Compute,
    input: &[f32],
    steps: usize,
    layer: &LstmLayer,
) -> Result<Vec<f32>> {
    if input.len() != steps * LSTM_HIDDEN {
        return Err(shape_error(format!(
            "LSTM input {} is not [{steps}, {LSTM_HIDDEN}]",
            input.len()
        )));
    }
    let gates = 4 * LSTM_HIDDEN;
    let mut input_gates = vec![0.0f32; steps * gates];
    compute.gemm_f32(
        steps,
        gates,
        LSTM_HIDDEN,
        input,
        &layer.weight_ih_t,
        Some(&layer.bias_ih),
        &mut input_gates,
    )?;

    let mut hidden = vec![0.0f32; LSTM_HIDDEN];
    let mut cell = vec![0.0f32; LSTM_HIDDEN];
    let mut recurrent_gates = vec![0.0f32; gates];
    let mut output = vec![0.0f32; steps * LSTM_HIDDEN];
    for step in 0..steps {
        compute.gemv_f32(
            gates,
            LSTM_HIDDEN,
            &layer.weight_hh,
            &hidden,
            Some(&layer.bias_hh),
            &mut recurrent_gates,
        )?;
        let offset = step * gates;
        for feature in 0..LSTM_HIDDEN {
            let input_gate = sigmoid(input_gates[offset + feature] + recurrent_gates[feature]);
            let forget_gate = sigmoid(
                input_gates[offset + LSTM_HIDDEN + feature]
                    + recurrent_gates[LSTM_HIDDEN + feature],
            );
            let candidate = (input_gates[offset + 2 * LSTM_HIDDEN + feature]
                + recurrent_gates[2 * LSTM_HIDDEN + feature])
                .tanh();
            let output_gate = sigmoid(
                input_gates[offset + 3 * LSTM_HIDDEN + feature]
                    + recurrent_gates[3 * LSTM_HIDDEN + feature],
            );
            cell[feature] = forget_gate * cell[feature] + input_gate * candidate;
            hidden[feature] = output_gate * cell[feature].tanh();
            output[step * LSTM_HIDDEN + feature] = hidden[feature];
        }
    }
    Ok(output)
}

fn conv_transpose1d(
    compute: &Compute,
    input: &[f32],
    input_length: usize,
    conv: &ConvTranspose1d,
    stride: usize,
) -> Result<(Vec<f32>, usize)> {
    if input.len() != conv.input * input_length || input_length == 0 {
        return Err(shape_error(format!(
            "ConvTranspose1d input {} is not [{} channels, {input_length} samples]",
            input.len(),
            conv.input
        )));
    }
    let time_major = channel_to_time_major(input, conv.input, input_length)?;
    let columns = conv.output * conv.kernel;
    let mut products = vec![0.0f32; input_length * columns];
    compute.gemm_f32(
        input_length,
        columns,
        conv.input,
        &time_major,
        &conv.weight_flat,
        None,
        &mut products,
    )?;
    let output_length = (input_length - 1)
        .checked_mul(stride)
        .and_then(|value| value.checked_add(conv.kernel))
        .ok_or_else(|| shape_error("ConvTranspose1d output length overflow"))?;
    let mut output = vec![0.0f32; conv.output * output_length];
    for channel in 0..conv.output {
        output[channel * output_length..(channel + 1) * output_length].fill(conv.bias[channel]);
    }
    for time in 0..input_length {
        for channel in 0..conv.output {
            for kernel in 0..conv.kernel {
                output[channel * output_length + time * stride + kernel] +=
                    products[time * columns + channel * conv.kernel + kernel];
            }
        }
    }
    Ok((output, output_length))
}

fn add_cropped_skip(
    input: &mut [f32],
    skip: &[f32],
    channels: usize,
    input_length: usize,
) -> Result<()> {
    if input.len() != channels * input_length || skip.len() % channels != 0 {
        return Err(shape_error("decoder/skip channel layout mismatch"));
    }
    let skip_length = skip.len() / channels;
    if skip_length < input_length {
        return Err(shape_error(format!(
            "encoder skip length {skip_length} is shorter than decoder length {input_length}"
        )));
    }
    for channel in 0..channels {
        for time in 0..input_length {
            input[channel * input_length + time] += skip[channel * skip_length + time];
        }
    }
    Ok(())
}

fn channel_to_time_major(input: &[f32], channels: usize, length: usize) -> Result<Vec<f32>> {
    if input.len() != channels * length {
        return Err(shape_error("channel-major transpose shape mismatch"));
    }
    let mut output = vec![0.0f32; input.len()];
    for channel in 0..channels {
        for time in 0..length {
            output[time * channels + channel] = input[channel * length + time];
        }
    }
    Ok(output)
}

fn time_to_channel_major(input: &[f32], length: usize, channels: usize) -> Result<Vec<f32>> {
    if input.len() != length * channels {
        return Err(shape_error("time-major transpose shape mismatch"));
    }
    let mut output = vec![0.0f32; input.len()];
    for time in 0..length {
        for channel in 0..channels {
            output[channel * length + time] = input[time * channels + channel];
        }
    }
    Ok(output)
}

fn relu_in_place(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

fn sinc_kernel() -> Vec<f32> {
    let taps = 2 * RESAMPLE_ZEROS;
    let window_length = 4 * RESAMPLE_ZEROS + 1;
    (0..taps)
        .map(|index| {
            let window_index = 2 * index + 1;
            let window =
                0.5 * (1.0 - (2.0 * PI * window_index as f32 / (window_length - 1) as f32).cos());
            let time = (index as f32 - (RESAMPLE_ZEROS as f32 - 0.5)) * PI;
            let sinc = if time == 0.0 { 1.0 } else { time.sin() / time };
            sinc * window
        })
        .collect()
}

fn upsample2(input: &[f32], channels: usize) -> Vec<f32> {
    let length = input.len() / channels;
    let kernel = sinc_kernel();
    let mut output = vec![0.0f32; channels * 2 * length];
    for channel in 0..channels {
        let source = &input[channel * length..(channel + 1) * length];
        let filtered = padded_cross_correlation(source, &kernel, RESAMPLE_ZEROS);
        for time in 0..length {
            output[channel * 2 * length + 2 * time] = source[time];
            output[channel * 2 * length + 2 * time + 1] = filtered[time + 1];
        }
    }
    output
}

fn downsample2(input: &[f32], channels: usize) -> Vec<f32> {
    let input_length = input.len() / channels;
    let length = input_length.div_ceil(2);
    let kernel = sinc_kernel();
    let mut output = vec![0.0f32; channels * length];
    for channel in 0..channels {
        let source = &input[channel * input_length..(channel + 1) * input_length];
        let mut odd = vec![0.0f32; length];
        for time in 0..length {
            if 2 * time + 1 < input_length {
                odd[time] = source[2 * time + 1];
            }
        }
        let filtered = padded_cross_correlation(&odd, &kernel, RESAMPLE_ZEROS);
        for time in 0..length {
            output[channel * length + time] = 0.5 * (source[2 * time] + filtered[time]);
        }
    }
    output
}

fn padded_cross_correlation(input: &[f32], kernel: &[f32], padding: usize) -> Vec<f32> {
    let output_length = input.len() + 2 * padding - kernel.len() + 1;
    let mut output = vec![0.0f32; output_length];
    for (position, value) in output.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (tap, &coefficient) in kernel.iter().enumerate() {
            let source = position as isize + tap as isize - padding as isize;
            if source >= 0 && (source as usize) < input.len() {
                sum += input[source as usize] * coefficient;
            }
        }
        *value = sum;
    }
    output
}

fn shape_error(message: impl Into<String>) -> VokraError {
    VokraError::InvalidArgument(format!("facebook_denoiser: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_length_round_trips_the_five_level_geometry() {
        for samples in [2usize, 8, 257, 1_000, 16_000, 32_001] {
            let valid = valid_length(samples);
            assert!(valid >= samples);
            let mut encoded = valid * 4;
            for _ in 0..DEPTH {
                encoded = (encoded - KERNEL_SIZE) / STRIDE + 1;
            }
            let mut decoded = encoded;
            for _ in 0..DEPTH {
                decoded = (decoded - 1) * STRIDE + KERNEL_SIZE;
            }
            assert_eq!(decoded / 4, valid);
        }
    }

    #[test]
    fn official_sinc_resampler_preserves_lengths_and_phase_order() {
        let input = vec![1.0, -0.5, 0.25];
        let up = upsample2(&input, 1);
        assert_eq!(up.len(), 6);
        assert_eq!(up[0], input[0]);
        assert_eq!(up[2], input[1]);
        assert_eq!(up[4], input[2]);
        assert_eq!(downsample2(&up, 1).len(), input.len());
    }

    #[test]
    fn normalization_uses_sample_standard_deviation() {
        let (normalized, std) = normalize(&[-1.0, 1.0]).unwrap();
        assert!((std - 2.0f32.sqrt()).abs() < 1.0e-6);
        assert!((normalized[0] + 1.0 / (NORMALIZATION_FLOOR + std)).abs() < 1.0e-6);
    }
}
