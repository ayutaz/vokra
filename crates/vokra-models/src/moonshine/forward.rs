use std::f32::consts::E;

use vokra_core::{BackendKind, Result, VokraError};

use crate::compute::{Compute, HotOp};

use super::MoonshineConfig;
use super::weights::{Attention, DecoderLayer, EncoderLayer, Linear, MoonshineWeights};

pub(super) const HOT_OPS: &[HotOp] = &[
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Conv1d,
];

pub(super) fn generate(
    weights: &MoonshineWeights,
    config: &MoonshineConfig,
    backend: BackendKind,
    pcm: &[f32],
) -> Result<Vec<u32>> {
    generate_with_limit(weights, config, backend, pcm, config.max_positions)
}

pub(super) fn generate_with_limit(
    weights: &MoonshineWeights,
    config: &MoonshineConfig,
    backend: BackendKind,
    pcm: &[f32],
    max_positions: usize,
) -> Result<Vec<u32>> {
    if backend != BackendKind::Cpu {
        return Err(VokraError::UnsupportedOp(format!(
            "moonshine: backend {backend:?} is not wired for the composed attention path; CPU is required (no silent CPU fallback)"
        )));
    }
    let compute = Compute::for_backend(backend, HOT_OPS)?;
    let (encoder, encoder_len) = encode(weights, config, &compute, pcm)?;
    let mut ids = vec![config.decoder_start_token_id];
    while ids.len() < max_positions {
        let hidden = decode(weights, config, &compute, &ids, &encoder, encoder_len)?;
        let last = &hidden[(ids.len() - 1) * config.hidden_size..ids.len() * config.hidden_size];
        let mut logits = vec![0.0; config.vocab_size];
        compute.gemv_f32(
            config.vocab_size,
            config.hidden_size,
            &weights.embedding,
            last,
            None,
            &mut logits,
        )?;
        let next = logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32)
            .ok_or_else(|| VokraError::ModelLoad("moonshine: empty logits".into()))?;
        if next == config.eos_token_id {
            break;
        }
        ids.push(next);
    }
    Ok(ids.into_iter().skip(1).collect())
}

pub(super) fn encode(
    weights: &MoonshineWeights,
    config: &MoonshineConfig,
    compute: &Compute,
    pcm: &[f32],
) -> Result<(Vec<f32>, usize)> {
    let d = config.hidden_size;
    let n1 = conv_out_len(pcm.len(), 127, 64)?;
    let mut conv1 = vec![0.0; d * n1];
    compute.conv1d_f32(
        pcm,
        1,
        pcm.len(),
        &weights.conv1.w,
        d,
        127,
        None,
        64,
        0,
        &mut conv1,
    )?;
    conv1.iter_mut().for_each(|value| *value = value.tanh());
    group_norm(
        &mut conv1,
        d,
        n1,
        &weights.groupnorm_weight,
        &weights.groupnorm_bias,
    );

    let n2 = conv_out_len(n1, 7, 3)?;
    let mut conv2 = vec![0.0; 2 * d * n2];
    compute.conv1d_f32(
        &conv1,
        d,
        n1,
        &weights.conv2.w,
        2 * d,
        7,
        weights.conv2.b.as_deref(),
        3,
        0,
        &mut conv2,
    )?;
    let mut activated = vec![0.0; conv2.len()];
    compute.gelu_f32(&conv2, &mut activated)?;

    let n3 = conv_out_len(n2, 3, 2)?;
    let mut conv3 = vec![0.0; d * n3];
    compute.conv1d_f32(
        &activated,
        2 * d,
        n2,
        &weights.conv3.w,
        d,
        3,
        weights.conv3.b.as_deref(),
        2,
        0,
        &mut conv3,
    )?;
    let mut conv3_activated = vec![0.0; conv3.len()];
    compute.gelu_f32(&conv3, &mut conv3_activated)?;

    if n3 > config.max_positions {
        return Err(VokraError::InvalidArgument(format!(
            "moonshine: audio produces {n3} encoder positions, maximum is {}",
            config.max_positions
        )));
    }
    let mut hidden = vec![0.0; n3 * d];
    for channel in 0..d {
        for time in 0..n3 {
            hidden[time * d + channel] = conv3_activated[channel * n3 + time];
        }
    }
    for layer in &weights.encoder_layers {
        encoder_layer(compute, config, layer, &mut hidden, n3)?;
    }
    hidden = layer_norm(compute, &hidden, n3, d, &weights.encoder_norm)?;
    Ok((hidden, n3))
}

pub(super) fn decode(
    weights: &MoonshineWeights,
    config: &MoonshineConfig,
    compute: &Compute,
    ids: &[u32],
    encoder: &[f32],
    encoder_len: usize,
) -> Result<Vec<f32>> {
    let d = config.hidden_size;
    let mut hidden = Vec::with_capacity(ids.len() * d);
    for &id in ids {
        let id = id as usize;
        if id >= config.vocab_size {
            return Err(VokraError::ModelLoad(format!(
                "moonshine: generated token {id} exceeds vocabulary"
            )));
        }
        hidden.extend_from_slice(&weights.embedding[id * d..(id + 1) * d]);
    }
    for layer in &weights.decoder_layers {
        decoder_layer(
            compute,
            config,
            layer,
            &mut hidden,
            ids.len(),
            encoder,
            encoder_len,
        )?;
    }
    layer_norm(compute, &hidden, ids.len(), d, &weights.decoder_norm)
}

fn encoder_layer(
    compute: &Compute,
    config: &MoonshineConfig,
    layer: &EncoderLayer,
    hidden: &mut [f32],
    rows: usize,
) -> Result<()> {
    let d = config.hidden_size;
    let normalized = layer_norm(compute, hidden, rows, d, &layer.ln1)?;
    let attended = attention(
        compute,
        config,
        &layer.attn,
        &normalized,
        rows,
        &normalized,
        rows,
        false,
        true,
    )?;
    residual_add(hidden, &attended);
    let normalized = layer_norm(compute, hidden, rows, d, &layer.ln2)?;
    let projected = linear_rows(
        compute,
        &layer.fc1,
        &normalized,
        rows,
        d,
        config.intermediate_size,
    )?;
    let mut activated = vec![0.0; projected.len()];
    compute.gelu_f32(&projected, &mut activated)?;
    let projected = linear_rows(
        compute,
        &layer.fc2,
        &activated,
        rows,
        config.intermediate_size,
        d,
    )?;
    residual_add(hidden, &projected);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decoder_layer(
    compute: &Compute,
    config: &MoonshineConfig,
    layer: &DecoderLayer,
    hidden: &mut [f32],
    rows: usize,
    encoder: &[f32],
    encoder_rows: usize,
) -> Result<()> {
    let d = config.hidden_size;
    let normalized = layer_norm(compute, hidden, rows, d, &layer.ln1)?;
    let attended = attention(
        compute,
        config,
        &layer.self_attn,
        &normalized,
        rows,
        &normalized,
        rows,
        true,
        true,
    )?;
    residual_add(hidden, &attended);

    let normalized = layer_norm(compute, hidden, rows, d, &layer.ln2)?;
    let attended = attention(
        compute,
        config,
        &layer.cross_attn,
        &normalized,
        rows,
        encoder,
        encoder_rows,
        false,
        false,
    )?;
    residual_add(hidden, &attended);

    let normalized = layer_norm(compute, hidden, rows, d, &layer.ln3)?;
    let gated = linear_rows(
        compute,
        &layer.fc1,
        &normalized,
        rows,
        d,
        2 * config.intermediate_size,
    )?;
    let ff = config.intermediate_size;
    let mut activated = vec![0.0; rows * ff];
    for row in 0..rows {
        let src = &gated[row * 2 * ff..(row + 1) * 2 * ff];
        let dst = &mut activated[row * ff..(row + 1) * ff];
        for index in 0..ff {
            let gate = src[ff + index];
            dst[index] = src[index] * gate / (1.0 + E.powf(-gate));
        }
    }
    let projected = linear_rows(compute, &layer.fc2, &activated, rows, ff, d)?;
    residual_add(hidden, &projected);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn attention(
    compute: &Compute,
    config: &MoonshineConfig,
    weights: &Attention,
    queries: &[f32],
    query_rows: usize,
    keys_values: &[f32],
    key_rows: usize,
    causal: bool,
    rotary: bool,
) -> Result<Vec<f32>> {
    let d = config.hidden_size;
    let heads = config.attention_heads;
    let head_dim = d / heads;
    let mut q = linear_rows(compute, &weights.q, queries, query_rows, d, d)?;
    let mut k = linear_rows(compute, &weights.k, keys_values, key_rows, d, d)?;
    let v = linear_rows(compute, &weights.v, keys_values, key_rows, d, d)?;
    if rotary {
        apply_rope(
            &mut q,
            query_rows,
            heads,
            head_dim,
            config.rotary_dim,
            config.rope_theta,
        );
        apply_rope(
            &mut k,
            key_rows,
            heads,
            head_dim,
            config.rotary_dim,
            config.rope_theta,
        );
    }
    let mut context = vec![0.0; query_rows * d];
    let scale = (head_dim as f32).sqrt().recip();
    for query in 0..query_rows {
        let visible = if causal { query + 1 } else { key_rows };
        for head in 0..heads {
            let q_base = query * d + head * head_dim;
            let mut scores = vec![0.0; visible];
            for (key, score) in scores.iter_mut().enumerate() {
                let k_base = key * d + head * head_dim;
                *score = dot(&q[q_base..q_base + head_dim], &k[k_base..k_base + head_dim]) * scale;
            }
            let mut probabilities = vec![0.0; visible];
            compute.softmax_f32(&scores, &mut probabilities, 1, visible)?;
            for dim in 0..head_dim {
                let mut sum = 0.0;
                for key in 0..visible {
                    sum += probabilities[key] * v[key * d + head * head_dim + dim];
                }
                context[q_base + dim] = sum;
            }
        }
    }
    linear_rows(compute, &weights.o, &context, query_rows, d, d)
}

fn linear_rows(
    compute: &Compute,
    linear: &Linear,
    input: &[f32],
    rows: usize,
    input_size: usize,
    output_size: usize,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0; rows * output_size];
    for row in 0..rows {
        compute.gemv_f32(
            output_size,
            input_size,
            &linear.w,
            &input[row * input_size..(row + 1) * input_size],
            linear.b.as_deref(),
            &mut output[row * output_size..(row + 1) * output_size],
        )?;
    }
    Ok(output)
}

fn layer_norm(
    compute: &Compute,
    input: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
) -> Result<Vec<f32>> {
    let mut output = vec![0.0; input.len()];
    let bias = vec![0.0; cols];
    compute.layer_norm_f32(input, &mut output, rows, cols, weight, &bias, 1e-5)?;
    Ok(output)
}

fn group_norm(values: &mut [f32], channels: usize, time: usize, weight: &[f32], bias: &[f32]) {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f32>()
        / values.len() as f32;
    let inverse_std = (variance + 1e-5).sqrt().recip();
    for channel in 0..channels {
        for index in 0..time {
            let slot = channel * time + index;
            values[slot] = (values[slot] - mean) * inverse_std * weight[channel] + bias[channel];
        }
    }
}

fn apply_rope(
    values: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
) {
    let d = heads * head_dim;
    for position in 0..rows {
        for head in 0..heads {
            let base = position * d + head * head_dim;
            for pair in (0..rotary_dim).step_by(2) {
                let frequency = theta.powf(-(pair as f32) / rotary_dim as f32);
                let angle = position as f32 * frequency;
                let (sin, cos) = angle.sin_cos();
                let left = values[base + pair];
                let right = values[base + pair + 1];
                values[base + pair] = left * cos - right * sin;
                values[base + pair + 1] = right * cos + left * sin;
            }
        }
    }
}

fn conv_out_len(input: usize, kernel: usize, stride: usize) -> Result<usize> {
    if input < kernel {
        return Err(VokraError::InvalidArgument(format!(
            "moonshine: input length {input} is shorter than Conv1D kernel {kernel}"
        )));
    }
    Ok((input - kernel) / stride + 1)
}

fn residual_add(left: &mut [f32], right: &[f32]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_lengths_match_reference_formula() {
        assert_eq!(conv_out_len(16_000, 127, 64).unwrap(), 249);
        assert_eq!(conv_out_len(249, 7, 3).unwrap(), 81);
        assert_eq!(conv_out_len(81, 3, 2).unwrap(), 40);
    }

    #[test]
    fn rope_rotates_first_pair_only() {
        let mut values = vec![1.0, 0.0, 3.0, 4.0];
        apply_rope(&mut values, 1, 1, 4, 2, 10_000.0);
        assert_eq!(values, vec![1.0, 0.0, 3.0, 4.0]);
    }
}
