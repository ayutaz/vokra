//! Focal modulation compressor/decompressor forward.

use vokra_core::backend::BackendKind;
use vokra_core::{Result, VokraError};
use vokra_ops::{VocosAttrs, VocosIstftPadding, VocosWeights};

use crate::align::charsiu::{layer_norm_with_compute_inplace, linear_forward_with_compute};
use crate::compute::Compute;
use crate::vocos::decode_weights_with_compute;

use super::weights::{
    FocalBlockWeights, FocalDecoderWeights, FocalEncoderWeights, LinearWeights, ScaleWeights,
};
use super::{CODE_DIM, CODEBOOK_SIZE, FOCALCODEC_HOT_OPS};

const LAYER_NORM_EPS: f32 = 1e-5;
const VOCOS_DIM: usize = 512;
const VOCOS_INPUT: usize = 1_024;
const VOCOS_FFN: usize = 1_536;

pub(super) fn compress_features(
    features: &[f32],
    frames: usize,
    weights: &FocalEncoderWeights,
    factors: [usize; 3],
    compute: &Compute,
) -> Result<(Vec<f32>, usize)> {
    if features.len() != frames * 1_024 || frames == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "focalcodec: compressor input has {} values for {frames}x1024",
            features.len()
        )));
    }
    let dims = [1_024usize, 512, 256];
    let mut input_dim = 1_024;
    let mut time = frames;
    let mut hidden = features.to_vec();
    for index in 0..3 {
        let (scaled, scaled_frames) = downscale(
            &hidden,
            time,
            input_dim,
            dims[index],
            factors[index],
            &weights.scales[index],
            compute,
        )?;
        hidden = focal_block(
            &scaled,
            scaled_frames,
            dims[index],
            &weights.blocks[index],
            compute,
        )?;
        time = scaled_frames;
        input_dim = dims[index];
    }
    let latents = linear(&hidden, time, 256, &weights.out_proj, 13, compute)?;
    reject_non_finite("compressor output", &latents)?;
    Ok((latents, time))
}

pub(super) fn latents_to_tokens(latents: &[f32], frames: usize) -> Result<Vec<u32>> {
    if frames == 0 || latents.len() != frames * CODE_DIM {
        return Err(VokraError::InvalidArgument(format!(
            "focalcodec: BSQ latents have {} values, expected {}",
            latents.len(),
            frames * CODE_DIM
        )));
    }
    let mut tokens = Vec::with_capacity(frames);
    for row in latents.chunks_exact(CODE_DIM) {
        let mut token = 0u32;
        for &value in row {
            token = (token << 1) | u32::from(value > 0.0);
        }
        debug_assert!((token as usize) < CODEBOOK_SIZE);
        tokens.push(token);
    }
    Ok(tokens)
}

pub(super) fn decode_tokens(
    tokens: &[u32],
    weights: &FocalDecoderWeights,
    vocos: &VocosWeights,
    factors: [usize; 3],
    backend: BackendKind,
) -> Result<Vec<f32>> {
    let compute = Compute::for_backend(backend, FOCALCODEC_HOT_OPS)?;
    let code_value = 1.0 / (CODE_DIM as f32).sqrt();
    let mut codes = Vec::with_capacity(tokens.len() * CODE_DIM);
    for &token in tokens {
        for shift in (0..CODE_DIM).rev() {
            let bit = (token >> shift) & 1;
            codes.push(if bit == 0 { -code_value } else { code_value });
        }
    }
    let (features, frames) = decompress_codes(&codes, tokens.len(), weights, factors, &compute)?;
    let attrs = VocosAttrs {
        input_channels: VOCOS_INPUT,
        dim: VOCOS_DIM,
        intermediate_dim: VOCOS_FFN,
        num_layers: 8,
        num_conditions: 0,
        n_fft: 1_024,
        hop_length: 320,
        padding: VocosIstftPadding::Same,
    };
    let pcm = decode_weights_with_compute(&features, frames, vocos, &attrs, &compute)?;
    reject_non_finite("decoder PCM", &pcm)?;
    Ok(pcm)
}

fn decompress_codes(
    codes: &[f32],
    frames: usize,
    weights: &FocalDecoderWeights,
    factors: [usize; 3],
    compute: &Compute,
) -> Result<(Vec<f32>, usize)> {
    if frames == 0 || codes.len() != frames * CODE_DIM {
        return Err(VokraError::InvalidArgument(format!(
            "focalcodec: decompressor codes have {} values, expected {}",
            codes.len(),
            frames * CODE_DIM
        )));
    }
    let dims = [256usize, 512, 1_024];
    let outputs = [512usize, 1_024, 1_024];
    let mut time = frames;
    let mut hidden = linear(codes, time, CODE_DIM, &weights.in_proj, 256, compute)?;
    for index in 0..3 {
        hidden = focal_block(&hidden, time, dims[index], &weights.blocks[index], compute)?;
        hidden = upscale(
            &hidden,
            time,
            dims[index],
            outputs[index],
            factors[index],
            &weights.scales[index],
            compute,
        )?;
        time = time.checked_mul(factors[index]).ok_or_else(|| {
            VokraError::InvalidArgument("focalcodec: decompressor frame count overflow".to_owned())
        })?;
    }
    reject_non_finite("decompressor output", &hidden)?;
    Ok((frame_to_channel(&hidden, time, 1_024), time))
}

fn downscale(
    input: &[f32],
    frames: usize,
    input_dim: usize,
    output_dim: usize,
    factor: usize,
    weights: &ScaleWeights,
    compute: &Compute,
) -> Result<(Vec<f32>, usize)> {
    let pad = (factor - frames % factor) % factor;
    let padded_frames = frames + pad;
    let mut channel_major = vec![0.0f32; input_dim * padded_frames];
    for frame in 0..frames {
        for channel in 0..input_dim {
            channel_major[channel * padded_frames + frame] = input[frame * input_dim + channel];
        }
    }
    let output_frames = padded_frames / factor;
    let mut output = vec![0.0f32; output_dim * output_frames];
    compute.conv1d_f32(
        &channel_major,
        input_dim,
        padded_frames,
        &weights.conv.weight,
        output_dim,
        factor,
        Some(&weights.conv.bias),
        factor,
        0,
        &mut output,
    )?;
    let mut activated = vec![0.0f32; output.len()];
    compute.snake_activation_f32(
        &output,
        &weights.alpha,
        output_dim,
        output_frames,
        &mut activated,
    )?;
    Ok((
        channel_to_frame(&activated, output_dim, output_frames),
        output_frames,
    ))
}

fn upscale(
    input: &[f32],
    frames: usize,
    input_dim: usize,
    output_dim: usize,
    factor: usize,
    weights: &ScaleWeights,
    compute: &Compute,
) -> Result<Vec<f32>> {
    let channel_major = frame_to_channel(input, frames, input_dim);
    let mut activated = vec![0.0f32; channel_major.len()];
    compute.snake_activation_f32(
        &channel_major,
        &weights.alpha,
        input_dim,
        frames,
        &mut activated,
    )?;
    let activated = channel_to_frame(&activated, input_dim, frames);
    let output_frames = frames.checked_mul(factor).ok_or_else(|| {
        VokraError::InvalidArgument("focalcodec: upscale frame count overflow".to_owned())
    })?;
    let mut output = vec![0.0f32; output_frames * output_dim];
    let mut tap_weight = vec![0.0f32; input_dim * output_dim];
    let mut tap_output = vec![0.0f32; frames * output_dim];
    for tap in 0..factor {
        for input_channel in 0..input_dim {
            for output_channel in 0..output_dim {
                tap_weight[input_channel * output_dim + output_channel] = weights.conv.weight
                    [(input_channel * output_dim + output_channel) * factor + tap];
            }
        }
        compute.gemm_f32(
            frames,
            output_dim,
            input_dim,
            &activated,
            &tap_weight,
            Some(&weights.conv.bias),
            &mut tap_output,
        )?;
        for frame in 0..frames {
            let destination = (frame * factor + tap) * output_dim;
            let source = frame * output_dim;
            output[destination..destination + output_dim]
                .copy_from_slice(&tap_output[source..source + output_dim]);
        }
    }
    Ok(output)
}

fn focal_block(
    input: &[f32],
    frames: usize,
    dim: usize,
    weights: &FocalBlockWeights,
    compute: &Compute,
) -> Result<Vec<f32>> {
    if input.len() != frames * dim {
        return Err(VokraError::InvalidArgument(format!(
            "focalcodec: focal block input has {} values, expected {}",
            input.len(),
            frames * dim
        )));
    }
    let mut normalized = input.to_vec();
    norm(
        &mut normalized,
        frames,
        dim,
        &weights.modulation_norm,
        compute,
    )?;
    let projected = linear(
        &normalized,
        frames,
        dim,
        &weights.modulation.in_proj,
        2 * dim + 3,
        compute,
    )?;
    let mut query = vec![0.0f32; frames * dim];
    let mut context = vec![0.0f32; dim * frames];
    let mut gates = vec![0.0f32; frames * 3];
    for frame in 0..frames {
        let source = frame * (2 * dim + 3);
        query[frame * dim..(frame + 1) * dim].copy_from_slice(&projected[source..source + dim]);
        for channel in 0..dim {
            context[channel * frames + frame] = projected[source + dim + channel];
        }
        gates[frame * 3..frame * 3 + 3]
            .copy_from_slice(&projected[source + 2 * dim..source + 2 * dim + 3]);
    }

    let mut context_all = vec![0.0f32; dim * frames];
    for level in 0..2 {
        let kernel = 7 + 2 * level;
        let depthwise = &weights.modulation.depthwise[level];
        let mut convolved = vec![0.0f32; context.len()];
        compute.grouped_conv1d_f32(
            &context,
            dim,
            frames,
            &depthwise.weight,
            dim,
            kernel,
            Some(&depthwise.bias),
            1,
            kernel / 2,
            dim,
            &mut convolved,
        )?;
        let mut activated = vec![0.0f32; convolved.len()];
        compute.gelu_f32(&convolved, &mut activated)?;
        context = activated;
        for channel in 0..dim {
            for frame in 0..frames {
                let index = channel * frames + frame;
                context_all[index] += context[index] * gates[frame * 3 + level];
            }
        }
    }

    let mut global = vec![0.0f32; dim];
    for channel in 0..dim {
        global[channel] = context[channel * frames..(channel + 1) * frames]
            .iter()
            .copied()
            .sum::<f32>()
            / frames as f32;
    }
    let mut global_activated = vec![0.0f32; dim];
    compute.gelu_f32(&global, &mut global_activated)?;
    for channel in 0..dim {
        for frame in 0..frames {
            context_all[channel * frames + frame] +=
                global_activated[channel] * gates[frame * 3 + 2];
        }
    }

    let mut modulator = vec![0.0f32; context_all.len()];
    compute.conv1d_f32(
        &context_all,
        dim,
        frames,
        &weights.modulation.context_proj.weight,
        dim,
        1,
        Some(&weights.modulation.context_proj.bias),
        1,
        0,
        &mut modulator,
    )?;
    let mut modulated = vec![0.0f32; frames * dim];
    for frame in 0..frames {
        for channel in 0..dim {
            modulated[frame * dim + channel] =
                query[frame * dim + channel] * modulator[channel * frames + frame];
        }
    }
    let projected = linear(
        &modulated,
        frames,
        dim,
        &weights.modulation.out_proj,
        dim,
        compute,
    )?;
    let mut hidden: Vec<f32> = input
        .iter()
        .zip(projected)
        .map(|(residual, value)| residual + value)
        .collect();

    normalized.copy_from_slice(&hidden);
    norm(
        &mut normalized,
        frames,
        dim,
        &weights.feed_forward_norm,
        compute,
    )?;
    let intermediate = linear(
        &normalized,
        frames,
        dim,
        &weights.feed_forward_in,
        4 * dim,
        compute,
    )?;
    let mut activated = vec![0.0f32; intermediate.len()];
    compute.gelu_f32(&intermediate, &mut activated)?;
    let output = linear(
        &activated,
        frames,
        4 * dim,
        &weights.feed_forward_out,
        dim,
        compute,
    )?;
    for (value, residual) in hidden.iter_mut().zip(output) {
        *value += residual;
    }
    Ok(hidden)
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
    weights: &super::weights::NormWeights,
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
    use crate::compute::Compute;
    use crate::focalcodec::weights::{ConvWeights, ScaleWeights};

    #[test]
    fn bsq_is_msb_first_and_spherical() {
        let latents = [
            1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
        ];
        assert_eq!(
            latents_to_tokens(&latents, 1).unwrap(),
            vec![0b1_0101_0101_0101]
        );
    }

    #[test]
    fn kernel_equals_stride_transpose_interleaves_taps() {
        let weights = ScaleWeights {
            conv: ConvWeights {
                // [in=1, out=1, kernel=2]
                weight: vec![2.0, 3.0],
                bias: vec![0.5],
            },
            alpha: vec![1.0],
        };
        let compute = Compute::for_backend(BackendKind::Cpu, FOCALCODEC_HOT_OPS).unwrap();
        let input = [0.0, 0.0];
        let output = upscale(&input, 2, 1, 1, 2, &weights, &compute).unwrap();
        // Snake(0)=0, so each tap receives just the ConvTranspose bias.
        assert_eq!(output, vec![0.5, 0.5, 0.5, 0.5]);
    }
}
