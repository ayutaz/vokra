//! Official ClearerVoice-Studio MossFormer2-SS-16K forward.

use vokra_core::{Result, VokraError};

use crate::compute::Compute;

use super::weights::{
    AttentionLayer, Conv1d, FfConv, FfNorm, FsmnCore, GatedFsmnLayer, Linear, Mossformer2Weights,
    Norm,
};
use super::{
    ENCODER_CHANNELS, ENCODER_KERNEL, ENCODER_STRIDE, GROUP_SIZE, OUTPUT_STREAMS, QUERY_KEY_DIM,
    SAMPLE_RATE,
};

const GROUP_NORM_EPS: f32 = 1.0e-8;
const LAYER_NORM_EPS: f32 = 1.0e-5;
const FINAL_NORM_EPS: f32 = 1.0e-6;
const INSTANCE_NORM_EPS: f32 = 1.0e-5;
const SCALE_NORM_EPS: f32 = 1.0e-5;
const DECODE_SECONDS: usize = 2;

#[derive(Debug, Clone)]
struct Sequence {
    /// Row-major `[length, channels]`.
    data: Vec<f32>,
    length: usize,
    channels: usize,
}

#[derive(Debug, Clone)]
struct Channels {
    /// Channel-major `[channels, length]`.
    data: Vec<f32>,
    channels: usize,
    length: usize,
}

pub(super) fn separate(
    compute: &Compute,
    weights: &Mossformer2Weights,
    pcm: &[f32],
) -> Result<Vec<Vec<f32>>> {
    // `MossFormer2_SS_16K.yaml`: one_time_decode_length=2,
    // decode_window=2; decode.py uses a 75%-window stride and discards half
    // of each overlap at both sides.
    let original_length = pcm.len();
    let window = SAMPLE_RATE as usize * DECODE_SECONDS;
    let stride = window * 3 / 4;
    let segmented = original_length > window;
    let input_rms = rms(pcm)?;
    let mut padded = pcm.to_vec();
    if original_length < window {
        padded.resize(window, 0.0);
    } else if original_length < window + stride {
        padded.resize(window + stride, 0.0);
    } else {
        let remainder = (original_length - window) % stride;
        if remainder != 0 {
            // Preserve the released ClearerVoice decode.py expression, even
            // though it pads by more than the conventional `stride-rem`.
            let padding = original_length
                .checked_sub(((original_length - window) / stride) * stride)
                .ok_or_else(|| {
                    VokraError::InvalidArgument(
                        "mossformer2-ss-16k: decode padding underflow".to_owned(),
                    )
                })?;
            let padded_length = original_length.checked_add(padding).ok_or_else(|| {
                VokraError::InvalidArgument(
                    "mossformer2-ss-16k: decode padding length overflow".to_owned(),
                )
            })?;
            padded.resize(padded_length, 0.0);
        }
    }

    let mut outputs = if segmented {
        let give_up = (window - stride) / 2;
        let mut accumulated = vec![vec![0.0f32; padded.len()]; OUTPUT_STREAMS];
        let mut start = 0usize;
        while start + window <= padded.len() {
            let chunk = separate_core(compute, weights, &padded[start..start + window])?;
            for stream in 0..OUTPUT_STREAMS {
                let (source_start, source_end, destination_start) = if start == 0 {
                    (0, window - give_up, 0)
                } else {
                    (give_up, window - give_up, start + give_up)
                };
                let count = source_end - source_start;
                accumulated[stream][destination_start..destination_start + count]
                    .copy_from_slice(&chunk[stream][source_start..source_end]);
            }
            start = start.checked_add(stride).ok_or_else(|| {
                VokraError::InvalidArgument(
                    "mossformer2-ss-16k: segmented decode index overflow".to_owned(),
                )
            })?;
        }
        accumulated
    } else {
        separate_core(compute, weights, &padded)?
    };

    for (stream, output) in outputs.iter_mut().enumerate() {
        let output_rms = rms(output).map_err(|_| {
            VokraError::InvalidArgument(format!(
                "mossformer2-ss-16k: output stream {stream} has zero/non-finite RMS"
            ))
        })?;
        let scale = input_rms / output_rms;
        for value in output.iter_mut() {
            *value *= scale;
        }
        output.truncate(original_length);
    }
    Ok(outputs)
}

pub(super) fn separate_core(
    compute: &Compute,
    weights: &Mossformer2Weights,
    pcm: &[f32],
) -> Result<Vec<Vec<f32>>> {
    let input = Channels {
        data: pcm.to_vec(),
        channels: 1,
        length: pcm.len(),
    };
    let mut encoded = conv1d(compute, &input, &weights.encoder, ENCODER_STRIDE, 0, 1)?;
    let mut activated = vec![0.0f32; encoded.data.len()];
    compute.relu_f32(&encoded.data, &mut activated)?;
    encoded.data = activated;
    let masks = mask_net(compute, weights, &encoded)?;
    if masks.len() != OUTPUT_STREAMS {
        return Err(VokraError::ModelLoad(format!(
            "mossformer2-ss-16k: mask net emitted {} streams, expected {OUTPUT_STREAMS}",
            masks.len()
        )));
    }

    let mut outputs = Vec::with_capacity(OUTPUT_STREAMS);
    for mask in masks {
        if mask.data.len() != encoded.data.len() {
            return Err(VokraError::ModelLoad(
                "mossformer2-ss-16k: mask/encoder shape mismatch".to_owned(),
            ));
        }
        let separated = Channels {
            data: encoded
                .data
                .iter()
                .zip(mask.data)
                .map(|(feature, gain)| feature * gain)
                .collect(),
            channels: ENCODER_CHANNELS,
            length: encoded.length,
        };
        let mut decoded = conv_transpose1d(
            compute,
            &separated,
            &weights.decoder,
            1,
            ENCODER_KERNEL,
            ENCODER_STRIDE,
        )?;
        decoded.resize(pcm.len(), 0.0);
        decoded.truncate(pcm.len());
        outputs.push(decoded);
    }
    Ok(outputs)
}

fn mask_net(
    compute: &Compute,
    weights: &Mossformer2Weights,
    encoded: &Channels,
) -> Result<Vec<Channels>> {
    let mask = &weights.mask;
    let normalized = group_norm(compute, encoded, &mask.input_norm, GROUP_NORM_EPS)?;
    let mut hidden = conv1d(compute, &normalized, &mask.input_projection, 1, 0, 1)?;
    add_sinusoidal_position(&mut hidden, &mask.position_inv_freq, mask.position_scale)?;
    let computation_residual = hidden.clone();
    let mut sequence = channels_to_sequence(&hidden);
    if mask.attention.len() != mask.fsmn.len() {
        return Err(VokraError::ModelLoad(
            "mossformer2-ss-16k: attention/FSMN layer count mismatch".to_owned(),
        ));
    }
    let rotary_freqs = rotary_frequencies(&mask.rotary_freqs)?;
    for (attention, fsmn) in mask.attention.iter().zip(&mask.fsmn) {
        sequence = flash_attention(compute, &sequence, attention, rotary_freqs)?;
        sequence = gated_fsmn(compute, &sequence, fsmn)?;
    }
    sequence = layer_norm_sequence(compute, &sequence, &mask.final_norm, FINAL_NORM_EPS)?;
    hidden = sequence_to_channels(&sequence);
    hidden = group_norm(compute, &hidden, &mask.intra_norm, GROUP_NORM_EPS)?;
    add_in_place(&mut hidden.data, &computation_residual.data)?;
    prelu_scalar(&mut hidden.data, mask.output_slope);
    let projected = conv1d(compute, &hidden, &mask.speaker_projection, 1, 0, 1)?;
    if projected.channels != OUTPUT_STREAMS * ENCODER_CHANNELS {
        return Err(VokraError::ModelLoad(format!(
            "mossformer2-ss-16k: speaker projection has {} channels",
            projected.channels
        )));
    }

    let mut masks = Vec::with_capacity(OUTPUT_STREAMS);
    for stream in 0..OUTPUT_STREAMS {
        let start = stream * ENCODER_CHANNELS * projected.length;
        let end = start + ENCODER_CHANNELS * projected.length;
        let lane = Channels {
            data: projected.data[start..end].to_vec(),
            channels: ENCODER_CHANNELS,
            length: projected.length,
        };
        let mut value = conv1d(compute, &lane, &mask.output, 1, 0, 1)?;
        let gate = conv1d(compute, &lane, &mask.output_gate, 1, 0, 1)?;
        for (value, gate) in value.data.iter_mut().zip(gate.data) {
            *value = value.tanh() * sigmoid(gate);
        }
        let mut decoded = conv1d(compute, &value, &mask.mask_projection, 1, 0, 1)?;
        let mut relu = vec![0.0f32; decoded.data.len()];
        compute.relu_f32(&decoded.data, &mut relu)?;
        decoded.data = relu;
        masks.push(decoded);
    }
    Ok(masks)
}

fn rotary_frequencies(frequencies: &[f32]) -> Result<&[f32]> {
    // `rotary_embedding_torch==0.8.3` registers `freqs` as a parameter. One
    // RotaryEmbedding instance is shared by all 24 FLASH layers, so PyTorch
    // serializes it only under layer 0 and every layer consumes these values.
    if frequencies.len() != 16 || frequencies.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(
            "mossformer2-ss-16k: invalid rotary frequency contract".to_owned(),
        ));
    }
    Ok(frequencies)
}

fn flash_attention(
    compute: &Compute,
    input: &Sequence,
    weights: &AttentionLayer,
    rotary_freqs: &[f32],
) -> Result<Sequence> {
    let residual = input.clone();
    let shifted = shift_first_half(input)?;
    let hidden = ff_conv(compute, &shifted, &weights.to_hidden)?;
    if hidden.channels != 2_048 {
        return Err(VokraError::ModelLoad(
            "mossformer2-ss-16k: to_hidden output is not 2048".to_owned(),
        ));
    }
    let mut value = Sequence {
        data: vec![0.0; input.length * 1_024],
        length: input.length,
        channels: 1_024,
    };
    let mut gate = value.clone();
    for row in 0..input.length {
        let source = row * 2_048;
        let destination = row * 1_024;
        value.data[destination..destination + 1_024]
            .copy_from_slice(&hidden.data[source..source + 1_024]);
        gate.data[destination..destination + 1_024]
            .copy_from_slice(&hidden.data[source + 1_024..source + 2_048]);
    }
    let qk = ff_conv(compute, &shifted, &weights.to_qk)?;
    if qk.channels != QUERY_KEY_DIM
        || weights.qk_gamma.len() != 4 * QUERY_KEY_DIM
        || weights.qk_beta.len() != 4 * QUERY_KEY_DIM
    {
        return Err(VokraError::ModelLoad(
            "mossformer2-ss-16k: q/k affine shape mismatch".to_owned(),
        ));
    }
    let mut heads = Vec::with_capacity(4);
    for head in 0..4 {
        let mut data = vec![0.0f32; qk.data.len()];
        let gamma = &weights.qk_gamma[head * QUERY_KEY_DIM..(head + 1) * QUERY_KEY_DIM];
        let beta = &weights.qk_beta[head * QUERY_KEY_DIM..(head + 1) * QUERY_KEY_DIM];
        for row in 0..qk.length {
            for channel in 0..QUERY_KEY_DIM {
                let index = row * QUERY_KEY_DIM + channel;
                data[index] = qk.data[index] * gamma[channel] + beta[channel];
            }
        }
        let mut head = Sequence {
            data,
            length: qk.length,
            channels: QUERY_KEY_DIM,
        };
        rotary_in_place(&mut head, rotary_freqs)?;
        heads.push(head);
    }
    let (attention_value, attention_gate) = grouped_attention(
        compute, &heads[0], &heads[1], &heads[2], &heads[3], &value, &gate,
    )?;
    let mut combined = Sequence {
        data: vec![0.0f32; input.length * 1_024],
        length: input.length,
        channels: 1_024,
    };
    for index in 0..combined.data.len() {
        combined.data[index] = attention_gate.data[index]
            * value.data[index]
            * sigmoid(attention_value.data[index] * gate.data[index]);
    }
    let mut output = ff_conv(compute, &combined, &weights.to_out)?;
    add_in_place(&mut output.data, &residual.data)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn grouped_attention(
    compute: &Compute,
    quad_q: &Sequence,
    lin_q: &Sequence,
    quad_k: &Sequence,
    lin_k: &Sequence,
    value: &Sequence,
    gate: &Sequence,
) -> Result<(Sequence, Sequence)> {
    let length = quad_q.length;
    if length == 0
        || [lin_q, quad_k, lin_k]
            .iter()
            .any(|sequence| sequence.length != length || sequence.channels != QUERY_KEY_DIM)
        || value.length != length
        || gate.length != length
        || value.channels != 1_024
        || gate.channels != 1_024
    {
        return Err(VokraError::InvalidArgument(
            "mossformer2-ss-16k: attention shape mismatch".to_owned(),
        ));
    }
    let groups = length.div_ceil(GROUP_SIZE);
    let padded = groups.checked_mul(GROUP_SIZE).ok_or_else(|| {
        VokraError::InvalidArgument("mossformer2-ss-16k: attention padding overflow".to_owned())
    })?;
    let mut out_value = Sequence {
        data: vec![0.0f32; length * 1_024],
        length,
        channels: 1_024,
    };
    let mut out_gate = out_value.clone();

    for group in 0..groups {
        let first = group * GROUP_SIZE;
        let valid = (length - first).min(GROUP_SIZE);
        let mut q = vec![0.0f32; GROUP_SIZE * QUERY_KEY_DIM];
        let mut k_t = vec![0.0f32; QUERY_KEY_DIM * GROUP_SIZE];
        let mut v = vec![0.0f32; GROUP_SIZE * 1_024];
        let mut u = vec![0.0f32; GROUP_SIZE * 1_024];
        for row in 0..valid {
            let source_q = (first + row) * QUERY_KEY_DIM;
            q[row * QUERY_KEY_DIM..(row + 1) * QUERY_KEY_DIM]
                .copy_from_slice(&quad_q.data[source_q..source_q + QUERY_KEY_DIM]);
            for channel in 0..QUERY_KEY_DIM {
                k_t[channel * GROUP_SIZE + row] = quad_k.data[source_q + channel];
            }
            let source_v = (first + row) * 1_024;
            v[row * 1_024..(row + 1) * 1_024]
                .copy_from_slice(&value.data[source_v..source_v + 1_024]);
            u[row * 1_024..(row + 1) * 1_024]
                .copy_from_slice(&gate.data[source_v..source_v + 1_024]);
        }
        let mut similarity = vec![0.0f32; GROUP_SIZE * GROUP_SIZE];
        compute.gemm_f32(
            GROUP_SIZE,
            GROUP_SIZE,
            QUERY_KEY_DIM,
            &q,
            &k_t,
            None,
            &mut similarity,
        )?;
        for score in &mut similarity {
            *score = (*score / GROUP_SIZE as f32).max(0.0).powi(2);
        }
        let mut group_value = vec![0.0f32; GROUP_SIZE * 1_024];
        let mut group_gate = vec![0.0f32; GROUP_SIZE * 1_024];
        compute.gemm_f32(
            GROUP_SIZE,
            1_024,
            GROUP_SIZE,
            &similarity,
            &v,
            None,
            &mut group_value,
        )?;
        compute.gemm_f32(
            GROUP_SIZE,
            1_024,
            GROUP_SIZE,
            &similarity,
            &u,
            None,
            &mut group_gate,
        )?;
        for row in 0..valid {
            let destination = (first + row) * 1_024;
            out_value.data[destination..destination + 1_024]
                .copy_from_slice(&group_value[row * 1_024..(row + 1) * 1_024]);
            out_gate.data[destination..destination + 1_024]
                .copy_from_slice(&group_gate[row * 1_024..(row + 1) * 1_024]);
        }
    }

    let mut linear_k_t = vec![0.0f32; QUERY_KEY_DIM * padded];
    let mut padded_value = vec![0.0f32; padded * 1_024];
    let mut padded_gate = vec![0.0f32; padded * 1_024];
    let mut padded_q = vec![0.0f32; padded * QUERY_KEY_DIM];
    for row in 0..length {
        let q_start = row * QUERY_KEY_DIM;
        padded_q[q_start..q_start + QUERY_KEY_DIM]
            .copy_from_slice(&lin_q.data[q_start..q_start + QUERY_KEY_DIM]);
        for channel in 0..QUERY_KEY_DIM {
            linear_k_t[channel * padded + row] = lin_k.data[q_start + channel];
        }
        let value_start = row * 1_024;
        padded_value[value_start..value_start + 1_024]
            .copy_from_slice(&value.data[value_start..value_start + 1_024]);
        padded_gate[value_start..value_start + 1_024]
            .copy_from_slice(&gate.data[value_start..value_start + 1_024]);
    }
    let mut kv = vec![0.0f32; QUERY_KEY_DIM * 1_024];
    let mut ku = vec![0.0f32; QUERY_KEY_DIM * 1_024];
    compute.gemm_f32(
        QUERY_KEY_DIM,
        1_024,
        padded,
        &linear_k_t,
        &padded_value,
        None,
        &mut kv,
    )?;
    compute.gemm_f32(
        QUERY_KEY_DIM,
        1_024,
        padded,
        &linear_k_t,
        &padded_gate,
        None,
        &mut ku,
    )?;
    let divisor = length as f32;
    for value in kv.iter_mut().chain(&mut ku) {
        *value /= divisor;
    }
    let mut linear_value = vec![0.0f32; padded * 1_024];
    let mut linear_gate = vec![0.0f32; padded * 1_024];
    compute.gemm_f32(
        padded,
        1_024,
        QUERY_KEY_DIM,
        &padded_q,
        &kv,
        None,
        &mut linear_value,
    )?;
    compute.gemm_f32(
        padded,
        1_024,
        QUERY_KEY_DIM,
        &padded_q,
        &ku,
        None,
        &mut linear_gate,
    )?;
    for row in 0..length {
        let start = row * 1_024;
        for channel in 0..1_024 {
            out_value.data[start + channel] += linear_value[start + channel];
            out_gate.data[start + channel] += linear_gate[start + channel];
        }
    }
    Ok((out_value, out_gate))
}

fn gated_fsmn(compute: &Compute, input: &Sequence, weights: &GatedFsmnLayer) -> Result<Sequence> {
    let residual = input.clone();
    let mut projected = conv1d(
        compute,
        &sequence_to_channels(input),
        &weights.input_conv,
        1,
        0,
        1,
    )?;
    prelu_scalar(&mut projected.data, weights.input_slope);
    let normalized = layer_norm_sequence(
        compute,
        &channels_to_sequence(&projected),
        &weights.norm1,
        LAYER_NORM_EPS,
    )?;
    let value = ff_conv(compute, &normalized, &weights.to_u)?;
    let gate = ff_conv(compute, &normalized, &weights.to_v)?;
    let fsmn = fsmn_core(compute, &value, &weights.fsmn)?;
    let mut gated = normalized.clone();
    for index in 0..gated.data.len() {
        gated.data[index] += gate.data[index] * fsmn.data[index];
    }
    let gated = layer_norm_sequence(compute, &gated, &weights.norm2, LAYER_NORM_EPS)?;
    let projected = conv1d(
        compute,
        &sequence_to_channels(&gated),
        &weights.output_conv,
        1,
        0,
        1,
    )?;
    let mut output = channels_to_sequence(&projected);
    add_in_place(&mut output.data, &residual.data)?;
    Ok(output)
}

fn fsmn_core(compute: &Compute, input: &Sequence, weights: &FsmnCore) -> Result<Sequence> {
    let residual = input.clone();
    let hidden = linear(compute, input, &weights.linear)?;
    let mut relu = vec![0.0f32; hidden.data.len()];
    compute.relu_f32(&hidden.data, &mut relu)?;
    let hidden = Sequence {
        data: relu,
        length: hidden.length,
        channels: hidden.channels,
    };
    let projected = linear(compute, &hidden, &weights.project)?;
    let mut skip = sequence_to_channels(&projected);
    let mut final_output = None;
    for (index, stage) in weights.dense.iter().enumerate() {
        let mut output = conv1d(
            compute,
            &skip,
            &stage.conv,
            1,
            (stage.conv.kernel - 1) * stage.dilation / 2,
            stage.dilation,
        )?;
        instance_norm(compute, &mut output, &stage.norm)?;
        prelu_channels(&mut output, &stage.slope)?;
        if index + 1 == weights.dense.len() {
            final_output = Some(output);
        } else {
            skip = concat_channels(&output, &skip)?;
        }
    }
    let mut output = channels_to_sequence(&final_output.ok_or_else(|| {
        VokraError::ModelLoad("mossformer2-ss-16k: FSMN dense net has no stages".to_owned())
    })?);
    add_in_place(&mut output.data, &residual.data)?;
    Ok(output)
}

fn ff_conv(compute: &Compute, input: &Sequence, weights: &FfConv) -> Result<Sequence> {
    let normalized = match &weights.norm {
        FfNorm::Scale { gain } => scale_norm(compute, input, *gain)?,
        FfNorm::Layer(norm) => layer_norm_sequence(compute, input, norm, LAYER_NORM_EPS)?,
    };
    let projected = linear(compute, &normalized, &weights.linear)?;
    let mut activated = vec![0.0f32; projected.data.len()];
    compute.silu_f32(&projected.data, &mut activated)?;
    let activated = Sequence {
        data: activated,
        length: projected.length,
        channels: projected.channels,
    };
    let convolved = conv1d(
        compute,
        &sequence_to_channels(&activated),
        &weights.depthwise,
        1,
        8,
        1,
    )?;
    let mut output = channels_to_sequence(&convolved);
    add_in_place(&mut output.data, &activated.data)?;
    Ok(output)
}

fn linear(compute: &Compute, input: &Sequence, weights: &Linear) -> Result<Sequence> {
    if input.channels != weights.input {
        return Err(VokraError::InvalidArgument(format!(
            "mossformer2-ss-16k: linear input {} != {}",
            input.channels, weights.input
        )));
    }
    let mut output = vec![0.0f32; input.length * weights.output];
    compute.gemm_f32(
        input.length,
        weights.output,
        weights.input,
        &input.data,
        &weights.weight_t,
        weights.bias.as_deref(),
        &mut output,
    )?;
    Ok(Sequence {
        data: output,
        length: input.length,
        channels: weights.output,
    })
}

fn conv1d(
    compute: &Compute,
    input: &Channels,
    weights: &Conv1d,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Channels> {
    if input.channels != weights.input || stride == 0 || dilation == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "mossformer2-ss-16k: Conv1D input/groups mismatch ({} channels, expected {}, stride={stride}, dilation={dilation})",
            input.channels, weights.input
        )));
    }
    let effective_kernel = (weights.kernel - 1)
        .checked_mul(dilation)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "mossformer2-ss-16k: Conv1D effective kernel overflow".to_owned(),
            )
        })?;
    let padded_length = input.length.checked_add(2 * padding).ok_or_else(|| {
        VokraError::InvalidArgument("mossformer2-ss-16k: Conv1D padded length overflow".to_owned())
    })?;
    if padded_length < effective_kernel {
        return Err(VokraError::InvalidArgument(format!(
            "mossformer2-ss-16k: Conv1D input {} with padding {padding} is shorter than kernel {effective_kernel}",
            input.length
        )));
    }
    let output_length = (padded_length - effective_kernel) / stride + 1;
    let expanded;
    let weight = if dilation == 1 {
        weights.weight.as_slice()
    } else {
        let inputs_per_group = weights.input / weights.groups;
        let mut value = vec![0.0f32; weights.output * inputs_per_group * effective_kernel];
        for output in 0..weights.output {
            for inner in 0..inputs_per_group {
                for tap in 0..weights.kernel {
                    let source = (output * inputs_per_group + inner) * weights.kernel + tap;
                    let destination =
                        (output * inputs_per_group + inner) * effective_kernel + tap * dilation;
                    value[destination] = weights.weight[source];
                }
            }
        }
        expanded = value;
        &expanded
    };
    let mut output = vec![0.0f32; weights.output * output_length];
    if weights.groups == 1 {
        compute.conv1d_f32(
            &input.data,
            input.channels,
            input.length,
            weight,
            weights.output,
            effective_kernel,
            weights.bias.as_deref(),
            stride,
            padding,
            &mut output,
        )?;
    } else {
        compute.grouped_conv1d_f32(
            &input.data,
            input.channels,
            input.length,
            weight,
            weights.output,
            effective_kernel,
            weights.bias.as_deref(),
            stride,
            padding,
            weights.groups,
            &mut output,
        )?;
    }
    Ok(Channels {
        data: output,
        channels: weights.output,
        length: output_length,
    })
}

fn conv_transpose1d(
    compute: &Compute,
    input: &Channels,
    weight: &[f32],
    output_channels: usize,
    kernel: usize,
    stride: usize,
) -> Result<Vec<f32>> {
    if input.length == 0 || stride == 0 || kernel == 0 {
        return Err(VokraError::InvalidArgument(
            "mossformer2-ss-16k: invalid ConvTranspose1D extent".to_owned(),
        ));
    }
    let expanded_length = (input.length - 1)
        .checked_mul(stride)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "mossformer2-ss-16k: ConvTranspose1D length overflow".to_owned(),
            )
        })?;
    let mut expanded_input = vec![0.0f32; input.channels * expanded_length];
    for channel in 0..input.channels {
        for frame in 0..input.length {
            expanded_input[channel * expanded_length + frame * stride] =
                input.data[channel * input.length + frame];
        }
    }
    if weight.len() != input.channels * output_channels * kernel {
        return Err(VokraError::ModelLoad(
            "mossformer2-ss-16k: decoder weight shape mismatch".to_owned(),
        ));
    }
    let mut flipped = vec![0.0f32; output_channels * input.channels * kernel];
    for input_channel in 0..input.channels {
        for output_channel in 0..output_channels {
            for tap in 0..kernel {
                flipped[(output_channel * input.channels + input_channel) * kernel + tap] = weight
                    [(input_channel * output_channels + output_channel) * kernel + kernel
                        - 1
                        - tap];
            }
        }
    }
    let padding = kernel - 1;
    let output_length = expanded_length + 2 * padding - kernel + 1;
    let mut output = vec![0.0f32; output_channels * output_length];
    compute.conv1d_f32(
        &expanded_input,
        input.channels,
        expanded_length,
        &flipped,
        output_channels,
        kernel,
        None,
        1,
        padding,
        &mut output,
    )?;
    Ok(output)
}

fn layer_norm_sequence(
    compute: &Compute,
    input: &Sequence,
    norm: &Norm,
    eps: f32,
) -> Result<Sequence> {
    if norm.gamma.len() != input.channels || norm.beta.len() != input.channels {
        return Err(VokraError::ModelLoad(
            "mossformer2-ss-16k: LayerNorm affine shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0f32; input.data.len()];
    compute.layer_norm_f32(
        &input.data,
        &mut output,
        input.length,
        input.channels,
        &norm.gamma,
        &norm.beta,
        eps,
    )?;
    Ok(Sequence {
        data: output,
        length: input.length,
        channels: input.channels,
    })
}

fn group_norm(compute: &Compute, input: &Channels, norm: &Norm, eps: f32) -> Result<Channels> {
    if norm.gamma.len() != input.channels || norm.beta.len() != input.channels {
        return Err(VokraError::ModelLoad(
            "mossformer2-ss-16k: GroupNorm affine shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0f32; input.data.len()];
    compute.group_norm_f32(
        &input.data,
        &mut output,
        input.channels,
        input.length,
        &norm.gamma,
        &norm.beta,
        eps,
    )?;
    Ok(Channels {
        data: output,
        channels: input.channels,
        length: input.length,
    })
}

fn instance_norm(compute: &Compute, input: &mut Channels, norm: &Norm) -> Result<()> {
    if norm.gamma.len() != input.channels || norm.beta.len() != input.channels {
        return Err(VokraError::ModelLoad(
            "mossformer2-ss-16k: InstanceNorm affine shape mismatch".to_owned(),
        ));
    }
    let unit_gamma = vec![1.0f32; input.length];
    let zero_beta = vec![0.0f32; input.length];
    let mut normalized = vec![0.0f32; input.data.len()];
    compute.layer_norm_f32(
        &input.data,
        &mut normalized,
        input.channels,
        input.length,
        &unit_gamma,
        &zero_beta,
        INSTANCE_NORM_EPS,
    )?;
    for channel in 0..input.channels {
        let start = channel * input.length;
        for value in &mut normalized[start..start + input.length] {
            *value = *value * norm.gamma[channel] + norm.beta[channel];
        }
    }
    input.data = normalized;
    Ok(())
}

fn scale_norm(compute: &Compute, input: &Sequence, gain: f32) -> Result<Sequence> {
    let mut output = vec![0.0f32; input.data.len()];
    compute.scale_norm_f32(
        &input.data,
        &mut output,
        input.length,
        input.channels,
        gain,
        SCALE_NORM_EPS,
    )?;
    Ok(Sequence {
        data: output,
        length: input.length,
        channels: input.channels,
    })
}

fn add_sinusoidal_position(input: &mut Channels, inv_freq: &[f32], scale: f32) -> Result<()> {
    if input.channels != 2 * inv_freq.len() || !scale.is_finite() {
        return Err(VokraError::ModelLoad(format!(
            "mossformer2-ss-16k: sinusoidal table {} does not match {} channels",
            inv_freq.len(),
            input.channels
        )));
    }
    for time in 0..input.length {
        for (frequency, &inverse) in inv_freq.iter().enumerate() {
            let angle = time as f32 * inverse;
            input.data[frequency * input.length + time] += angle.sin() * scale;
            input.data[(inv_freq.len() + frequency) * input.length + time] += angle.cos() * scale;
        }
    }
    Ok(())
}

fn rotary_in_place(input: &mut Sequence, frequencies: &[f32]) -> Result<()> {
    if input.channels != QUERY_KEY_DIM || frequencies.len() * 2 > input.channels {
        return Err(VokraError::ModelLoad(
            "mossformer2-ss-16k: rotary embedding shape mismatch".to_owned(),
        ));
    }
    for time in 0..input.length {
        let start = time * input.channels;
        for (frequency, &inverse) in frequencies.iter().enumerate() {
            let even = start + frequency * 2;
            let odd = even + 1;
            let angle = time as f32 * inverse;
            let cosine = angle.cos();
            let sine = angle.sin();
            let left = input.data[even];
            let right = input.data[odd];
            input.data[even] = left * cosine - right * sine;
            input.data[odd] = right * cosine + left * sine;
        }
    }
    Ok(())
}

fn shift_first_half(input: &Sequence) -> Result<Sequence> {
    if !input.channels.is_multiple_of(2) {
        return Err(VokraError::InvalidArgument(
            "mossformer2-ss-16k: token shift needs an even channel count".to_owned(),
        ));
    }
    let half = input.channels / 2;
    let mut output = input.clone();
    for time in 0..input.length {
        let destination = time * input.channels;
        if time == 0 {
            output.data[destination..destination + half].fill(0.0);
        } else {
            let source = (time - 1) * input.channels;
            output.data[destination..destination + half]
                .copy_from_slice(&input.data[source..source + half]);
        }
    }
    Ok(output)
}

fn channels_to_sequence(input: &Channels) -> Sequence {
    let mut data = vec![0.0f32; input.data.len()];
    for channel in 0..input.channels {
        for time in 0..input.length {
            data[time * input.channels + channel] = input.data[channel * input.length + time];
        }
    }
    Sequence {
        data,
        length: input.length,
        channels: input.channels,
    }
}

fn sequence_to_channels(input: &Sequence) -> Channels {
    let mut data = vec![0.0f32; input.data.len()];
    for time in 0..input.length {
        for channel in 0..input.channels {
            data[channel * input.length + time] = input.data[time * input.channels + channel];
        }
    }
    Channels {
        data,
        channels: input.channels,
        length: input.length,
    }
}

fn concat_channels(left: &Channels, right: &Channels) -> Result<Channels> {
    if left.length != right.length {
        return Err(VokraError::InvalidArgument(
            "mossformer2-ss-16k: dense skip length mismatch".to_owned(),
        ));
    }
    let mut data = left.data.clone();
    data.extend_from_slice(&right.data);
    Ok(Channels {
        data,
        channels: left.channels + right.channels,
        length: left.length,
    })
}

fn prelu_scalar(values: &mut [f32], slope: f32) {
    for value in values {
        if *value < 0.0 {
            *value *= slope;
        }
    }
}

fn prelu_channels(input: &mut Channels, slopes: &[f32]) -> Result<()> {
    if slopes.len() != input.channels {
        return Err(VokraError::ModelLoad(
            "mossformer2-ss-16k: channel PReLU shape mismatch".to_owned(),
        ));
    }
    for channel in 0..input.channels {
        let start = channel * input.length;
        prelu_scalar(
            &mut input.data[start..start + input.length],
            slopes[channel],
        );
    }
    Ok(())
}

fn add_in_place(destination: &mut [f32], source: &[f32]) -> Result<()> {
    if destination.len() != source.len() {
        return Err(VokraError::InvalidArgument(format!(
            "mossformer2-ss-16k: residual lengths {} and {} differ",
            destination.len(),
            source.len()
        )));
    }
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += source;
    }
    Ok(())
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn rms(values: &[f32]) -> Result<f32> {
    if values.is_empty() {
        return Err(VokraError::InvalidArgument(
            "mossformer2-ss-16k: RMS input is empty".to_owned(),
        ));
    }
    let mean_square = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    let value = mean_square.sqrt();
    if !value.is_finite() || value <= 0.0 {
        return Err(VokraError::InvalidArgument(
            "mossformer2-ss-16k: RMS is zero or non-finite".to_owned(),
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_shift_moves_only_the_first_half() {
        let input = Sequence {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            length: 2,
            channels: 4,
        };
        let output = shift_first_half(&input).expect("shift");
        assert_eq!(output.data, vec![0.0, 0.0, 3.0, 4.0, 1.0, 2.0, 7.0, 8.0]);
    }

    #[test]
    fn channel_sequence_roundtrip_is_exact() {
        let channels = Channels {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            channels: 2,
            length: 3,
        };
        assert_eq!(
            sequence_to_channels(&channels_to_sequence(&channels)).data,
            channels.data
        );
    }

    #[test]
    fn rotary_rotates_adjacent_pairs_and_leaves_tail() {
        let mut sequence = Sequence {
            data: (0..QUERY_KEY_DIM).map(|value| value as f32).collect(),
            length: 1,
            channels: QUERY_KEY_DIM,
        };
        let original = sequence.data.clone();
        rotary_in_place(&mut sequence, &[1.0; 16]).expect("rotary");
        assert_eq!(sequence.data, original, "position zero is the identity");
    }
}
