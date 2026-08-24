//! WavLM waveform encoder used by FocalCodec.

use vokra_core::backend::BackendKind;
use vokra_core::{Result, VokraError};

use crate::align::charsiu::{layer_norm_with_compute_inplace, linear_forward_with_compute};
use crate::compute::Compute;

use super::FOCALCODEC_HOT_OPS;
use super::focal::{compress_features, latents_to_tokens};
use super::weights::{
    FEATURE_DIM, FocalEncoderWeights, LinearWeights, NormWeights, RELATIVE_BUCKETS, WAVLM_DIM,
    WAVLM_FFN, WAVLM_HEAD_DIM, WAVLM_HEADS, WavLmAttentionWeights, WavLmBlockWeights, WavLmWeights,
};

const FEATURE_KERNELS: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
const FEATURE_STRIDES: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];
const LAYER_NORM_EPS: f32 = 1e-5;
const POSITION_KERNEL: usize = 128;
const POSITION_GROUPS: usize = 16;
const MAX_DISTANCE: usize = 800;

pub(super) fn encode_tokens(
    pcm: &[f32],
    wavlm: &WavLmWeights,
    compressor: &FocalEncoderWeights,
    factors: [usize; 3],
    backend: BackendKind,
) -> Result<Vec<u32>> {
    let compute = Compute::for_backend(backend, FOCALCODEC_HOT_OPS)?;
    let (features, frames) = encode_features(pcm, wavlm, &compute)?;
    let (latents, latent_frames) =
        compress_features(&features, frames, compressor, factors, &compute)?;
    latents_to_tokens(&latents, latent_frames)
}

fn encode_features(
    pcm: &[f32],
    weights: &WavLmWeights,
    compute: &Compute,
) -> Result<(Vec<f32>, usize)> {
    debug_assert_eq!(weights.unused_output_norm.weight.len(), WAVLM_DIM);
    debug_assert_eq!(weights.unused_output_norm.bias.len(), WAVLM_DIM);
    let mut channel_major = pcm.to_vec();
    let mut input_channels = 1usize;
    let mut time = pcm.len();
    for index in 0..FEATURE_KERNELS.len() {
        let kernel = FEATURE_KERNELS[index];
        let stride = FEATURE_STRIDES[index];
        let padded_time = time.max(kernel);
        if padded_time != time {
            let mut padded = vec![0.0f32; input_channels * padded_time];
            for channel in 0..input_channels {
                padded[channel * padded_time..channel * padded_time + time]
                    .copy_from_slice(&channel_major[channel * time..(channel + 1) * time]);
            }
            channel_major = padded;
        }
        let output_time = (padded_time - kernel) / stride + 1;
        let layer = &weights.feature_layers[index];
        let mut convolved = vec![0.0f32; FEATURE_DIM * output_time];
        compute.conv1d_f32(
            &channel_major,
            input_channels,
            padded_time,
            &layer.conv.weight,
            FEATURE_DIM,
            kernel,
            None,
            stride,
            0,
            &mut convolved,
        )?;
        let mut frame_major = channel_to_frame(&convolved, FEATURE_DIM, output_time);
        norm(
            &mut frame_major,
            output_time,
            FEATURE_DIM,
            &layer.norm,
            compute,
        )?;
        let mut activated = vec![0.0f32; frame_major.len()];
        compute.gelu_f32(&frame_major, &mut activated)?;
        channel_major = frame_to_channel(&activated, output_time, FEATURE_DIM);
        time = output_time;
        input_channels = FEATURE_DIM;
    }

    let mut hidden = channel_to_frame(&channel_major, FEATURE_DIM, time);
    norm(&mut hidden, time, FEATURE_DIM, &weights.input_norm, compute)?;
    hidden = linear(
        &hidden,
        time,
        FEATURE_DIM,
        &weights.feature_proj,
        WAVLM_DIM,
        compute,
    )?;

    let hidden_channels = frame_to_channel(&hidden, time, WAVLM_DIM);
    let positional_time = time + 1;
    let mut positional = vec![0.0f32; WAVLM_DIM * positional_time];
    compute.grouped_conv1d_f32(
        &hidden_channels,
        WAVLM_DIM,
        time,
        &weights.positional_conv.weight,
        WAVLM_DIM,
        POSITION_KERNEL,
        Some(&weights.positional_conv.bias),
        1,
        POSITION_KERNEL / 2,
        POSITION_GROUPS,
        &mut positional,
    )?;
    let mut trimmed = vec![0.0f32; WAVLM_DIM * time];
    for channel in 0..WAVLM_DIM {
        trimmed[channel * time..(channel + 1) * time].copy_from_slice(
            &positional[channel * positional_time..channel * positional_time + time],
        );
    }
    let mut positional_activated = vec![0.0f32; trimmed.len()];
    compute.gelu_f32(&trimmed, &mut positional_activated)?;
    let positional_activated = channel_to_frame(&positional_activated, WAVLM_DIM, time);
    for (value, position) in hidden.iter_mut().zip(positional_activated) {
        *value += position;
    }

    let relative_bias = relative_position_bias(time, &weights.relative_embedding)?;
    for block in &weights.blocks {
        transformer_block(&mut hidden, time, block, &relative_bias, compute)?;
    }
    reject_non_finite("WavLM output", &hidden)?;
    Ok((hidden, time))
}

fn transformer_block(
    hidden: &mut [f32],
    frames: usize,
    weights: &WavLmBlockWeights,
    relative_bias: &[f32],
    compute: &Compute,
) -> Result<()> {
    let mut normalized = hidden.to_vec();
    norm(
        &mut normalized,
        frames,
        WAVLM_DIM,
        &weights.attention_norm,
        compute,
    )?;
    let attention = attention(
        &normalized,
        frames,
        &weights.attention,
        relative_bias,
        compute,
    )?;
    let projected = linear(
        &attention,
        frames,
        WAVLM_DIM,
        &weights.attention.out,
        WAVLM_DIM,
        compute,
    )?;
    for (value, residual) in hidden.iter_mut().zip(projected) {
        *value += residual;
    }

    normalized.copy_from_slice(hidden);
    norm(
        &mut normalized,
        frames,
        WAVLM_DIM,
        &weights.feed_forward_norm,
        compute,
    )?;
    let intermediate = linear(
        &normalized,
        frames,
        WAVLM_DIM,
        &weights.feed_forward_in,
        WAVLM_FFN,
        compute,
    )?;
    let mut activated = vec![0.0f32; intermediate.len()];
    compute.gelu_f32(&intermediate, &mut activated)?;
    let output = linear(
        &activated,
        frames,
        WAVLM_FFN,
        &weights.feed_forward_out,
        WAVLM_DIM,
        compute,
    )?;
    for (value, residual) in hidden.iter_mut().zip(output) {
        *value += residual;
    }
    Ok(())
}

fn attention(
    input: &[f32],
    frames: usize,
    weights: &WavLmAttentionWeights,
    relative_bias: &[f32],
    compute: &Compute,
) -> Result<Vec<f32>> {
    let q = linear(input, frames, WAVLM_DIM, &weights.q, WAVLM_DIM, compute)?;
    let k = linear(input, frames, WAVLM_DIM, &weights.k, WAVLM_DIM, compute)?;
    let v = linear(input, frames, WAVLM_DIM, &weights.v, WAVLM_DIM, compute)?;

    let mut gates = vec![0.0f32; frames * WAVLM_HEADS];
    for frame in 0..frames {
        for head in 0..WAVLM_HEADS {
            let row = &input[frame * WAVLM_DIM + head * WAVLM_HEAD_DIM
                ..frame * WAVLM_DIM + (head + 1) * WAVLM_HEAD_DIM];
            let mut projected = [0.0f32; 8];
            for (out, projected_value) in projected.iter_mut().enumerate() {
                let mut value = weights.gru_bias[out];
                for (inner, row_value) in row.iter().copied().enumerate() {
                    value += row_value * weights.gru_weight[out * WAVLM_HEAD_DIM + inner];
                }
                *projected_value = value;
            }
            let gate_a = sigmoid(projected[..4].iter().copied().sum());
            let gate_b = sigmoid(projected[4..].iter().copied().sum());
            gates[frame * WAVLM_HEADS + head] =
                gate_a * (gate_b * weights.gru_const[head] - 1.0) + 2.0;
        }
    }

    let scale = 1.0 / (WAVLM_HEAD_DIM as f32).sqrt();
    let mut output = vec![0.0f32; frames * WAVLM_DIM];
    let mut q_head = vec![0.0f32; frames * WAVLM_HEAD_DIM];
    let mut k_head_t = vec![0.0f32; WAVLM_HEAD_DIM * frames];
    let mut v_head = vec![0.0f32; frames * WAVLM_HEAD_DIM];
    let mut scores = vec![0.0f32; frames * frames];
    let mut probabilities = vec![0.0f32; scores.len()];
    let mut head_output = vec![0.0f32; frames * WAVLM_HEAD_DIM];
    for head in 0..WAVLM_HEADS {
        for frame in 0..frames {
            let source = frame * WAVLM_DIM + head * WAVLM_HEAD_DIM;
            let destination = frame * WAVLM_HEAD_DIM;
            q_head[destination..destination + WAVLM_HEAD_DIM]
                .copy_from_slice(&q[source..source + WAVLM_HEAD_DIM]);
            v_head[destination..destination + WAVLM_HEAD_DIM]
                .copy_from_slice(&v[source..source + WAVLM_HEAD_DIM]);
            for inner in 0..WAVLM_HEAD_DIM {
                k_head_t[inner * frames + frame] = k[source + inner];
            }
        }
        compute.gemm_f32(
            frames,
            frames,
            WAVLM_HEAD_DIM,
            &q_head,
            &k_head_t,
            None,
            &mut scores,
        )?;
        for query in 0..frames {
            let gate = gates[query * WAVLM_HEADS + head];
            for key in 0..frames {
                let index = query * frames + key;
                scores[index] = scores[index] * scale
                    + gate * relative_bias[(head * frames + query) * frames + key];
            }
        }
        compute.softmax_f32(&scores, &mut probabilities, frames, frames)?;
        compute.gemm_f32(
            frames,
            WAVLM_HEAD_DIM,
            frames,
            &probabilities,
            &v_head,
            None,
            &mut head_output,
        )?;
        for frame in 0..frames {
            let source = frame * WAVLM_HEAD_DIM;
            let destination = frame * WAVLM_DIM + head * WAVLM_HEAD_DIM;
            output[destination..destination + WAVLM_HEAD_DIM]
                .copy_from_slice(&head_output[source..source + WAVLM_HEAD_DIM]);
        }
    }
    Ok(output)
}

fn relative_position_bias(frames: usize, embedding: &[f32]) -> Result<Vec<f32>> {
    if embedding.len() != RELATIVE_BUCKETS * WAVLM_HEADS {
        return Err(VokraError::ModelLoad(format!(
            "focalcodec: WavLM relative embedding has {} values, expected {}",
            embedding.len(),
            RELATIVE_BUCKETS * WAVLM_HEADS
        )));
    }
    let mut bias = vec![0.0f32; WAVLM_HEADS * frames * frames];
    for query in 0..frames {
        for key in 0..frames {
            let relative = key as isize - query as isize;
            let bucket = relative_bucket(relative);
            for head in 0..WAVLM_HEADS {
                bias[(head * frames + query) * frames + key] =
                    embedding[bucket * WAVLM_HEADS + head];
            }
        }
    }
    Ok(bias)
}

fn relative_bucket(relative: isize) -> usize {
    const HALF_BUCKETS: usize = RELATIVE_BUCKETS / 2;
    const MAX_EXACT: usize = HALF_BUCKETS / 2;
    let sign = usize::from(relative > 0) * HALF_BUCKETS;
    let distance = relative.unsigned_abs();
    let bucket = if distance < MAX_EXACT {
        distance
    } else {
        let scaled = ((distance as f32 / MAX_EXACT as f32).ln()
            / (MAX_DISTANCE as f32 / MAX_EXACT as f32).ln()
            * (HALF_BUCKETS - MAX_EXACT) as f32)
            + MAX_EXACT as f32;
        (scaled as usize).min(HALF_BUCKETS - 1)
    };
    sign + bucket
}

fn linear(
    input: &[f32],
    frames: usize,
    input_dim: usize,
    weights: &LinearWeights,
    output_dim: usize,
    compute: &Compute,
) -> Result<Vec<f32>> {
    linear_forward_with_compute(
        input,
        frames,
        input_dim,
        &weights.weight,
        &weights.bias,
        output_dim,
        compute,
    )
}

fn norm(
    input: &mut [f32],
    frames: usize,
    dim: usize,
    weights: &NormWeights,
    compute: &Compute,
) -> Result<()> {
    layer_norm_with_compute_inplace(
        input,
        frames,
        dim,
        &weights.weight,
        &weights.bias,
        LAYER_NORM_EPS,
        compute,
    )
}

fn frame_to_channel(input: &[f32], frames: usize, channels: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for frame in 0..frames {
        for channel in 0..channels {
            output[channel * frames + frame] = input[frame * channels + channel];
        }
    }
    output
}

fn channel_to_frame(input: &[f32], channels: usize, frames: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for channel in 0..channels {
        for frame in 0..frames {
            output[frame * channels + channel] = input[channel * frames + frame];
        }
    }
    output
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn reject_non_finite(label: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "focalcodec: {label} contains non-finite value {value} at index {index}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_bucket_matches_wavlm_boundaries() {
        assert_eq!(relative_bucket(0), 0);
        assert_eq!(relative_bucket(-79), 79);
        assert_eq!(relative_bucket(79), 239);
        assert_eq!(relative_bucket(-800), 159);
        assert_eq!(relative_bucket(800), 319);
        assert_eq!(relative_bucket(-80_000), 159);
    }

    #[test]
    fn sigmoid_is_stable_at_large_magnitude() {
        assert_eq!(sigmoid(1_000.0), 1.0);
        assert_eq!(sigmoid(-1_000.0), 0.0);
    }
}
