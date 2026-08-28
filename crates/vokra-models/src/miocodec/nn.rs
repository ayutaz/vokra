//! Native MioCodec token-to-waveform forward.
//!
//! Learned operations dispatch through [`Compute`]. Layout changes, RoPE,
//! local-attention masking, interpolation, residual additions and iSTFT
//! assembly are deterministic host glue; selecting Metal never substitutes a
//! CPU learned kernel.

use vokra_core::ir::graph::IstftAttrs;
use vokra_core::{Result, VokraError};
use vokra_ops::{Spectrogram, Xcodec2FsqAttrs, istft};

use crate::compute::Compute;

use super::weights::{
    AdaTransformer, AffineNorm, AffineTransformer, Attention, Conv1d, ConvTranspose1d, FeedForward,
    Linear, MioCodecWeights, ResnetBlock,
};
use super::{
    CONTENT_DIM, FSQ_LEVELS, HOP_LENGTH, ISTFT_BINS, N_FFT, ROPE_THETA, UPSAMPLE_TOTAL, WAVE_DIM,
};

const LAYER_NORM_EPS: f32 = 1.0e-5;
const GROUP_NORM_EPS: f32 = 1.0e-6;

pub(super) fn decode(
    compute: &Compute,
    weights: &MioCodecWeights,
    codes: &[u32],
    global_embedding: &[f32],
    target_samples: usize,
) -> Result<Vec<f32>> {
    let pre_upsample_frames = target_samples / HOP_LENGTH / UPSAMPLE_TOTAL;
    if pre_upsample_frames == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "miocodec: target_samples {target_samples} is too short; need at least {}",
            HOP_LENGTH * UPSAMPLE_TOTAL
        )));
    }

    let fsq = Xcodec2FsqAttrs {
        levels: FSQ_LEVELS.to_vec(),
        d_model: CONTENT_DIM,
    };
    let content = compute.xcodec2_fsq_f32(codes, codes.len(), Some(&weights.fsq_output), &fsq)?;
    let prenet = affine_transformer(compute, &content, codes.len(), &weights.prenet)?;
    let prenet_channels = frame_to_channel(&prenet, codes.len(), WAVE_DIM);
    let first_upsampled = conv_transpose(
        compute,
        &prenet_channels,
        codes.len(),
        &weights.first_upsample,
    )?;
    let first_frames = codes
        .len()
        .checked_mul(weights.first_upsample.stride)
        .ok_or_else(|| VokraError::InvalidArgument("miocodec: frame count overflow".to_owned()))?;
    let mut hidden = interpolate_linear_channels(
        &first_upsampled,
        WAVE_DIM,
        first_frames,
        pre_upsample_frames,
    )?;
    for block in &weights.prior {
        hidden = resnet(compute, &hidden, pre_upsample_frames, block)?;
    }

    let frame_major = channel_to_frame(&hidden, WAVE_DIM, pre_upsample_frames);
    let decoded = ada_transformer(
        compute,
        &frame_major,
        pre_upsample_frames,
        global_embedding,
        &weights.decoder,
    )?;
    hidden = frame_to_channel(&decoded, pre_upsample_frames, WAVE_DIM);
    for block in &weights.post {
        hidden = resnet(compute, &hidden, pre_upsample_frames, block)?;
    }

    let mut frames = pre_upsample_frames;
    for stage in &weights.upsampler.stages {
        hidden = conv_transpose(compute, &hidden, frames, &stage.transpose)?;
        frames = frames.checked_mul(stage.transpose.stride).ok_or_else(|| {
            VokraError::InvalidArgument("miocodec: upsampler frame count overflow".to_owned())
        })?;
        hidden = snake_beta(compute, &hidden, &stage.alpha, &stage.beta, frames)?;
        hidden = resnet(compute, &hidden, frames, &stage.resnet)?;
    }

    let frame_major = channel_to_frame(&hidden, 128, frames);
    let projected = linear(
        compute,
        &frame_major,
        frames,
        &weights.upsampler.output_projection,
    )?;
    let projected = frame_to_channel(&projected, frames, WAVE_DIM);
    let projected = snake_beta(
        compute,
        &projected,
        &weights.upsampler.output_alpha,
        &weights.upsampler.output_beta,
        frames,
    )?;
    let frame_major = channel_to_frame(&projected, WAVE_DIM, frames);
    let stft_parameters = linear(compute, &frame_major, frames, &weights.istft_projection)?;
    istft_head(&stft_parameters, frames)
}

fn affine_transformer(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    weights: &AffineTransformer,
) -> Result<Vec<f32>> {
    let mut hidden = input.to_vec();
    for block in &weights.layers {
        let normalized = layer_norm(compute, &hidden, frames, &block.attention_norm)?;
        let attended = attention(compute, &normalized, frames, &block.attention)?;
        add_residual(&mut hidden, &attended)?;
        let normalized = layer_norm(compute, &hidden, frames, &block.ffn_norm)?;
        let feed_forward = feed_forward(compute, &normalized, frames, &block.feed_forward)?;
        add_residual(&mut hidden, &feed_forward)?;
    }
    let hidden = layer_norm(compute, &hidden, frames, &weights.norm)?;
    linear(compute, &hidden, frames, &weights.output_projection)
}

fn ada_transformer(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    condition: &[f32],
    weights: &AdaTransformer,
) -> Result<Vec<f32>> {
    let mut condition_silu = vec![0.0f32; condition.len()];
    compute.silu_f32(condition, &mut condition_silu)?;
    let mut hidden = input.to_vec();
    for block in &weights.layers {
        let parameters = linear(compute, &condition_silu, 1, &block.attention_condition)?;
        let (shift, rest) = parameters.split_at(WAVE_DIM);
        let (scale, gate) = rest.split_at(WAVE_DIM);
        let normalized = adaptive_layer_norm(compute, &hidden, frames, shift, scale)?;
        let mut attended = attention(compute, &normalized, frames, &block.attention)?;
        gate_rows(&mut attended, frames, WAVE_DIM, gate)?;
        add_residual(&mut hidden, &attended)?;

        let parameters = linear(compute, &condition_silu, 1, &block.ffn_condition)?;
        let (shift, rest) = parameters.split_at(WAVE_DIM);
        let (scale, gate) = rest.split_at(WAVE_DIM);
        let normalized = adaptive_layer_norm(compute, &hidden, frames, shift, scale)?;
        let mut feed_forward = feed_forward(compute, &normalized, frames, &block.feed_forward)?;
        gate_rows(&mut feed_forward, frames, WAVE_DIM, gate)?;
        add_residual(&mut hidden, &feed_forward)?;
    }
    let parameters = linear(compute, &condition_silu, 1, &weights.final_condition)?;
    let (shift, scale) = parameters.split_at(WAVE_DIM);
    adaptive_layer_norm(compute, &hidden, frames, shift, scale)
}

fn attention(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    weights: &Attention,
) -> Result<Vec<f32>> {
    if weights.q.input != weights.q.output
        || weights.q.input != weights.k.input
        || weights.q.input != weights.k.output
        || weights.q.input != weights.v.input
        || weights.q.input != weights.v.output
        || weights.q.input != weights.out.input
        || weights.q.input != weights.out.output
        || weights.heads == 0
        || weights.q.input % weights.heads != 0
        || weights.window == 0
        || weights.window % 2 == 0
    {
        return Err(VokraError::ModelLoad(
            "miocodec: invalid attention dimensions".to_owned(),
        ));
    }
    let dim = weights.q.input;
    let head_dim = dim / weights.heads;
    let mut q = linear(compute, input, frames, &weights.q)?;
    let mut k = linear(compute, input, frames, &weights.k)?;
    let v = linear(compute, input, frames, &weights.v)?;
    apply_rope(&mut q, frames, weights.heads, head_dim)?;
    apply_rope(&mut k, frames, weights.heads, head_dim)?;

    let mut merged = vec![0.0f32; frames * dim];
    let scale = (head_dim as f32).sqrt().recip();
    let window_side = weights.window / 2;
    for head in 0..weights.heads {
        let mut q_head = vec![0.0f32; frames * head_dim];
        let mut k_transposed = vec![0.0f32; head_dim * frames];
        let mut v_head = vec![0.0f32; frames * head_dim];
        for frame in 0..frames {
            let source = frame * dim + head * head_dim;
            q_head[frame * head_dim..(frame + 1) * head_dim]
                .copy_from_slice(&q[source..source + head_dim]);
            v_head[frame * head_dim..(frame + 1) * head_dim]
                .copy_from_slice(&v[source..source + head_dim]);
            for inner in 0..head_dim {
                k_transposed[inner * frames + frame] = k[source + inner];
            }
        }
        let mut logits = vec![0.0f32; frames * frames];
        compute.gemm_f32(
            frames,
            frames,
            head_dim,
            &q_head,
            &k_transposed,
            None,
            &mut logits,
        )?;
        for query in 0..frames {
            for key in 0..frames {
                let value = &mut logits[query * frames + key];
                if query.abs_diff(key) > window_side {
                    *value = f32::NEG_INFINITY;
                } else {
                    *value *= scale;
                }
            }
        }
        let mut probabilities = vec![0.0f32; logits.len()];
        compute.softmax_f32(&logits, &mut probabilities, frames, frames)?;
        let mut context = vec![0.0f32; frames * head_dim];
        compute.gemm_f32(
            frames,
            head_dim,
            frames,
            &probabilities,
            &v_head,
            None,
            &mut context,
        )?;
        for frame in 0..frames {
            let target = frame * dim + head * head_dim;
            merged[target..target + head_dim]
                .copy_from_slice(&context[frame * head_dim..(frame + 1) * head_dim]);
        }
    }
    linear(compute, &merged, frames, &weights.out)
}

fn apply_rope(values: &mut [f32], frames: usize, heads: usize, head_dim: usize) -> Result<()> {
    if head_dim == 0 || head_dim % 2 != 0 || values.len() != frames * heads * head_dim {
        return Err(VokraError::InvalidArgument(
            "miocodec: RoPE shape mismatch".to_owned(),
        ));
    }
    let dim = heads * head_dim;
    for frame in 0..frames {
        for head in 0..heads {
            let start = frame * dim + head * head_dim;
            for pair in 0..head_dim / 2 {
                let exponent = (2 * pair) as f32 / head_dim as f32;
                let angle = frame as f32 / ROPE_THETA.powf(exponent);
                let cos = angle.cos();
                let sin = angle.sin();
                let offset = start + 2 * pair;
                let real = values[offset];
                let imaginary = values[offset + 1];
                values[offset] = real * cos - imaginary * sin;
                values[offset + 1] = imaginary * cos + real * sin;
            }
        }
    }
    Ok(())
}

fn feed_forward(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    weights: &FeedForward,
) -> Result<Vec<f32>> {
    let w1 = linear(compute, input, frames, &weights.w1)?;
    let w3 = linear(compute, input, frames, &weights.w3)?;
    let mut activated = vec![0.0f32; w1.len()];
    compute.silu_f32(&w1, &mut activated)?;
    for (value, gate) in activated.iter_mut().zip(w3) {
        *value *= gate;
    }
    linear(compute, &activated, frames, &weights.w2)
}

fn linear(compute: &Compute, input: &[f32], rows: usize, weights: &Linear) -> Result<Vec<f32>> {
    let input_elements = rows.checked_mul(weights.input).ok_or_else(|| {
        VokraError::InvalidArgument("miocodec: linear input size overflow".to_owned())
    })?;
    let weight_elements = weights
        .input
        .checked_mul(weights.output)
        .ok_or_else(|| VokraError::ModelLoad("miocodec: linear weight size overflow".to_owned()))?;
    if input.len() != input_elements
        || weights.weight_t.len() != weight_elements
        || weights
            .bias
            .as_ref()
            .is_some_and(|bias| bias.len() != weights.output)
    {
        return Err(VokraError::InvalidArgument(format!(
            "miocodec: linear shape mismatch: input {} (expected {input_elements}), weight {} (expected {weight_elements}), bias {:?} (expected {:?})",
            input.len(),
            weights.weight_t.len(),
            weights.bias.as_ref().map(Vec::len),
            weights.bias.as_ref().map(|_| weights.output),
        )));
    }
    let output_elements = rows.checked_mul(weights.output).ok_or_else(|| {
        VokraError::InvalidArgument("miocodec: linear output size overflow".to_owned())
    })?;
    let mut output = vec![0.0f32; output_elements];
    compute.gemm_f32(
        rows,
        weights.output,
        weights.input,
        input,
        &weights.weight_t,
        weights.bias.as_deref(),
        &mut output,
    )?;
    Ok(output)
}

fn layer_norm(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    weights: &AffineNorm,
) -> Result<Vec<f32>> {
    let dim = weights.weight.len();
    if weights.bias.len() != dim || input.len() != frames * dim {
        return Err(VokraError::InvalidArgument(
            "miocodec: affine LayerNorm shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0f32; input.len()];
    compute.layer_norm_f32(
        input,
        &mut output,
        frames,
        dim,
        &weights.weight,
        &weights.bias,
        LAYER_NORM_EPS,
    )?;
    Ok(output)
}

fn adaptive_layer_norm(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    shift: &[f32],
    scale: &[f32],
) -> Result<Vec<f32>> {
    if shift.len() != WAVE_DIM || scale.len() != WAVE_DIM || input.len() != frames * WAVE_DIM {
        return Err(VokraError::InvalidArgument(
            "miocodec: AdaLN shape mismatch".to_owned(),
        ));
    }
    let gamma = vec![1.0f32; WAVE_DIM];
    let beta = vec![0.0f32; WAVE_DIM];
    let mut output = vec![0.0f32; input.len()];
    compute.layer_norm_f32(
        input,
        &mut output,
        frames,
        WAVE_DIM,
        &gamma,
        &beta,
        LAYER_NORM_EPS,
    )?;
    for frame in 0..frames {
        for dim in 0..WAVE_DIM {
            let value = &mut output[frame * WAVE_DIM + dim];
            *value = *value * (1.0 + scale[dim]) + shift[dim];
        }
    }
    Ok(output)
}

fn gate_rows(values: &mut [f32], frames: usize, dim: usize, gate: &[f32]) -> Result<()> {
    if values.len() != frames * dim || gate.len() != dim {
        return Err(VokraError::InvalidArgument(
            "miocodec: AdaLN gate shape mismatch".to_owned(),
        ));
    }
    for row in values.chunks_exact_mut(dim) {
        for (value, &gate) in row.iter_mut().zip(gate) {
            *value *= gate;
        }
    }
    Ok(())
}

fn resnet(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    weights: &ResnetBlock,
) -> Result<Vec<f32>> {
    let channels = weights.conv1.input;
    let mut hidden = group_norm(
        compute,
        input,
        channels,
        frames,
        weights.groups,
        &weights.norm1,
    )?;
    hidden = silu(compute, &hidden)?;
    hidden = conv_same(compute, &hidden, frames, &weights.conv1)?;
    hidden = group_norm(
        compute,
        &hidden,
        channels,
        frames,
        weights.groups,
        &weights.norm2,
    )?;
    hidden = silu(compute, &hidden)?;
    hidden = conv_same(compute, &hidden, frames, &weights.conv2)?;
    add_residual(&mut hidden, input)?;
    Ok(hidden)
}

fn group_norm(
    compute: &Compute,
    input: &[f32],
    channels: usize,
    frames: usize,
    groups: usize,
    weights: &AffineNorm,
) -> Result<Vec<f32>> {
    if groups == 0
        || channels % groups != 0
        || input.len() != channels * frames
        || weights.weight.len() != channels
        || weights.bias.len() != channels
    {
        return Err(VokraError::InvalidArgument(
            "miocodec: GroupNorm shape mismatch".to_owned(),
        ));
    }
    let channels_per_group = channels / groups;
    let values_per_group = channels_per_group * frames;
    let mut output = vec![0.0f32; input.len()];
    for group in 0..groups {
        let values = group * values_per_group;
        let affine = group * channels_per_group;
        compute.group_norm_f32(
            &input[values..values + values_per_group],
            &mut output[values..values + values_per_group],
            channels_per_group,
            frames,
            &weights.weight[affine..affine + channels_per_group],
            &weights.bias[affine..affine + channels_per_group],
            GROUP_NORM_EPS,
        )?;
    }
    Ok(output)
}

fn conv_same(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    weights: &Conv1d,
) -> Result<Vec<f32>> {
    if input.len() != weights.input * frames {
        return Err(VokraError::InvalidArgument(
            "miocodec: Conv1d input shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0f32; weights.output * frames];
    compute.conv1d_f32(
        input,
        weights.input,
        frames,
        &weights.weight,
        weights.output,
        weights.kernel,
        Some(&weights.bias),
        1,
        weights.kernel / 2,
        &mut output,
    )?;
    Ok(output)
}

fn conv_transpose(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    weights: &ConvTranspose1d,
) -> Result<Vec<f32>> {
    if frames == 0 || input.len() != weights.input * frames {
        return Err(VokraError::InvalidArgument(
            "miocodec: ConvTranspose1d input shape mismatch".to_owned(),
        ));
    }
    let expanded_frames = (frames - 1)
        .checked_mul(weights.stride)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            VokraError::InvalidArgument("miocodec: ConvTranspose1d expansion overflow".to_owned())
        })?;
    let output_frames = (frames - 1)
        .checked_mul(weights.stride)
        .and_then(|value| value.checked_add(weights.kernel))
        .and_then(|value| value.checked_sub(2 * weights.padding))
        .ok_or_else(|| {
            VokraError::InvalidArgument("miocodec: ConvTranspose1d output overflow".to_owned())
        })?;
    let mut expanded = vec![0.0f32; weights.input * expanded_frames];
    for channel in 0..weights.input {
        for frame in 0..frames {
            expanded[channel * expanded_frames + frame * weights.stride] =
                input[channel * frames + frame];
        }
    }
    let conv_padding = weights
        .kernel
        .checked_sub(1 + weights.padding)
        .ok_or_else(|| {
            VokraError::ModelLoad("miocodec: invalid ConvTranspose padding".to_owned())
        })?;
    let mut output = vec![0.0f32; weights.output * output_frames];
    compute.conv1d_f32(
        &expanded,
        weights.input,
        expanded_frames,
        &weights.conv_weight,
        weights.output,
        weights.kernel,
        Some(&weights.bias),
        1,
        conv_padding,
        &mut output,
    )?;
    Ok(output)
}

fn interpolate_linear_channels(
    input: &[f32],
    channels: usize,
    input_frames: usize,
    output_frames: usize,
) -> Result<Vec<f32>> {
    if input_frames == 0 || output_frames == 0 || input.len() != channels * input_frames {
        return Err(VokraError::InvalidArgument(
            "miocodec: interpolation shape mismatch".to_owned(),
        ));
    }
    if input_frames == output_frames {
        return Ok(input.to_vec());
    }
    let scale = input_frames as f64 / output_frames as f64;
    let mut output = vec![0.0f32; channels * output_frames];
    for channel in 0..channels {
        for target in 0..output_frames {
            let source =
                ((target as f64 + 0.5) * scale - 0.5).clamp(0.0, (input_frames - 1) as f64);
            let left = source.floor() as usize;
            let right = (left + 1).min(input_frames - 1);
            let fraction = (source - left as f64) as f32;
            let a = input[channel * input_frames + left];
            let b = input[channel * input_frames + right];
            output[channel * output_frames + target] = a + (b - a) * fraction;
        }
    }
    Ok(output)
}

fn snake_beta(
    compute: &Compute,
    input: &[f32],
    alpha: &[f32],
    beta: &[f32],
    frames: usize,
) -> Result<Vec<f32>> {
    let channels = alpha.len();
    if beta.len() != channels || input.len() != channels * frames {
        return Err(VokraError::InvalidArgument(
            "miocodec: SnakeBeta shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0f32; input.len()];
    compute.snake_beta_f32(input, alpha, beta, channels, frames, &mut output)?;
    Ok(output)
}

fn silu(compute: &Compute, input: &[f32]) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; input.len()];
    compute.silu_f32(input, &mut output)?;
    Ok(output)
}

fn istft_head(parameters: &[f32], frames: usize) -> Result<Vec<f32>> {
    if parameters.len() != frames * (N_FFT + 2) {
        return Err(VokraError::InvalidArgument(
            "miocodec: iSTFT projection shape mismatch".to_owned(),
        ));
    }
    let mut re = vec![0.0f32; frames * ISTFT_BINS];
    let mut im = vec![0.0f32; frames * ISTFT_BINS];
    for frame in 0..frames {
        for bin in 0..ISTFT_BINS {
            let magnitude = parameters[frame * (N_FFT + 2) + bin].exp().min(100.0);
            let phase = parameters[frame * (N_FFT + 2) + ISTFT_BINS + bin];
            re[frame * ISTFT_BINS + bin] = magnitude * phase.cos();
            im[frame * ISTFT_BINS + bin] = magnitude * phase.sin();
        }
    }
    let spectrogram = Spectrogram {
        frames,
        bins: ISTFT_BINS,
        re,
        im,
    };
    let mut attrs = IstftAttrs::new(N_FFT, HOP_LENGTH);
    attrs.center = false;
    let pcm = istft(&spectrogram, &attrs)?;
    let trim = (N_FFT - HOP_LENGTH) / 2;
    if pcm.len() < 2 * trim {
        return Err(VokraError::InvalidArgument(format!(
            "miocodec: iSTFT output {} is shorter than same-padding trim {}",
            pcm.len(),
            2 * trim
        )));
    }
    let pcm = pcm[trim..pcm.len() - trim].to_vec();
    let expected = frames
        .checked_mul(HOP_LENGTH)
        .ok_or_else(|| VokraError::InvalidArgument("miocodec: PCM size overflow".to_owned()))?;
    if pcm.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "miocodec: iSTFT emitted {} samples, expected {expected}",
            pcm.len()
        )));
    }
    Ok(pcm)
}

fn add_residual(output: &mut [f32], residual: &[f32]) -> Result<()> {
    if output.len() != residual.len() {
        return Err(VokraError::InvalidArgument(
            "miocodec: residual shape mismatch".to_owned(),
        ));
    }
    for (output, &residual) in output.iter_mut().zip(residual) {
        *output += residual;
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_pixel_linear_interpolation_matches_torch_geometry() {
        let input = [0.0, 10.0];
        let got = interpolate_linear_channels(&input, 1, 2, 4).unwrap();
        assert_eq!(got, vec![0.0, 2.5, 7.5, 10.0]);
    }

    #[test]
    fn frame_channel_transposes_round_trip() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let channels = frame_to_channel(&input, 2, 3);
        assert_eq!(channels, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(channel_to_frame(&channels, 3, 2), input);
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let mut values = [1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut values, 1, 1, 4).unwrap();
        assert_eq!(values, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn zero_inserted_conv_matches_convtranspose_geometry() {
        let weights = ConvTranspose1d {
            conv_weight: vec![30.0, 20.0, 10.0],
            bias: vec![1.0],
            input: 1,
            output: 1,
            kernel: 3,
            stride: 2,
            padding: 1,
        };
        let got = conv_transpose(&Compute::cpu(), &[1.0, 2.0], 2, &weights).unwrap();
        assert_eq!(got, vec![21.0, 51.0, 41.0]);
    }

    #[test]
    fn same_padded_istft_emits_one_hop_per_frame() {
        let frames = 9;
        let got = istft_head(&vec![0.0; frames * (N_FFT + 2)], frames).unwrap();
        assert_eq!(got.len(), frames * HOP_LENGTH);
        assert!(got.iter().all(|sample| sample.is_finite()));
    }
}
