//! Official TIGER forward. Learned operations dispatch through Compute;
//! indexing, STFT/iSTFT, interpolation and scalar activations are host glue.

use vokra_core::ir::graph::{IstftAttrs, StftAttrs};
use vokra_core::{Result, VokraError};
use vokra_ops::{Spectrogram, istft, stft};

use crate::compute::Compute;

use super::weights::{
    Attention, AttentionProjection, CoreWeights, DepthNorm, GroupedConv, Injection, Mlp, Norm,
    Pointwise, PointwiseNorm, TigerWeights, UConv,
};
use super::{ITERATIONS, TigerVariant};

const GLOBAL_NORM_EPS: f32 = 1.0e-8;
const LAYER_NORM_EPS: f32 = 1.0e-5;

#[derive(Debug, Clone)]
struct Sequences {
    /// Batch-major, then channel-major [batch, channels, length].
    data: Vec<f32>,
    batch: usize,
    channels: usize,
    length: usize,
}

#[derive(Debug, Clone)]
struct Grid {
    /// [batch, channels, rows, columns].
    data: Vec<f32>,
    batch: usize,
    channels: usize,
    rows: usize,
    columns: usize,
}

pub(super) fn separate(
    compute: &Compute,
    variant: TigerVariant,
    weights: &TigerWeights,
    pcm: &[f32],
) -> Result<Vec<Vec<f32>>> {
    match (variant, weights) {
        (TigerVariant::Speech, TigerWeights::Speech(core)) => {
            separate_core(compute, variant, core, pcm)
        }
        (
            TigerVariant::Dnr,
            TigerWeights::Dnr {
                dialog,
                effect,
                music,
            },
        ) => Ok(vec![
            chunked_selected(compute, dialog, pcm, 2)?,
            chunked_selected(compute, effect, pcm, 1)?,
            chunked_selected(compute, music, pcm, 0)?,
        ]),
        _ => Err(VokraError::ModelLoad(
            "tiger: variant and weight topology disagree".to_owned(),
        )),
    }
}

fn chunked_selected(
    compute: &Compute,
    core: &CoreWeights,
    pcm: &[f32],
    selected_source: usize,
) -> Result<Vec<f32>> {
    let variant = TigerVariant::Dnr;
    let sample_rate = variant.sample_rate() as usize;
    let session = sample_rate * 12;
    let hop = sample_rate * 4;
    let overlap_pad = session - hop;
    let mut padded = vec![0.0f32; overlap_pad];
    padded.extend_from_slice(pcm);
    padded.resize(padded.len() + overlap_pad, 0.0);

    let num_sessions = (padded.len() - session) / hop + 2;
    let mut accumulated = vec![0.0f32; padded.len()];
    for index in 0..num_sessions {
        let start = index * hop;
        let available = padded.len().saturating_sub(start).min(session);
        let mut segment = vec![0.0f32; session];
        if available > 0 {
            segment[..available].copy_from_slice(&padded[start..start + available]);
        }
        let outputs = separate_core(compute, variant, core, &segment)?;
        let selected = outputs.get(selected_source).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "tiger-dnr: core returned {} streams, source {selected_source} requested",
                outputs.len()
            ))
        })?;
        let writable = available
            .min(selected.len())
            .min(accumulated.len().saturating_sub(start));
        for offset in 0..writable {
            accumulated[start + offset] += selected[offset];
        }
    }
    let crop_start = overlap_pad;
    let crop_end = crop_start.checked_add(pcm.len()).ok_or_else(|| {
        VokraError::InvalidArgument("tiger-dnr: output crop length overflow".to_owned())
    })?;
    if crop_end > accumulated.len() {
        return Err(VokraError::InvalidArgument(
            "tiger-dnr: chunk accumulation is shorter than the requested crop".to_owned(),
        ));
    }
    Ok(accumulated[crop_start..crop_end]
        .iter()
        .map(|value| value / 3.0)
        .collect())
}

fn separate_core(
    compute: &Compute,
    variant: TigerVariant,
    weights: &CoreWeights,
    pcm: &[f32],
) -> Result<Vec<Vec<f32>>> {
    let n_fft = variant.n_fft();
    let hop = variant.hop_length();
    let spectrogram = stft(pcm, &StftAttrs::new(n_fft, hop))?;
    let widths = variant.band_widths();
    let channels = variant.feature_channels();
    if spectrogram.bins != n_fft / 2 + 1
        || widths.iter().sum::<usize>() != spectrogram.bins
        || weights.bands.len() != widths.len()
        || weights.masks.len() != widths.len()
    {
        return Err(VokraError::ModelLoad(
            "tiger: frontend band topology disagrees with the checkpoint".to_owned(),
        ));
    }

    let mut features = Grid {
        data: vec![0.0; channels * widths.len() * spectrogram.frames],
        batch: 1,
        channels,
        rows: widths.len(),
        columns: spectrogram.frames,
    };
    let mut first_bin = 0;
    for (band, (&width, band_weights)) in widths.iter().zip(&weights.bands).enumerate() {
        let mut packed = vec![0.0f32; 2 * width * spectrogram.frames];
        for bin in 0..width {
            for frame in 0..spectrogram.frames {
                let source = frame * spectrogram.bins + first_bin + bin;
                packed[bin * spectrogram.frames + frame] = spectrogram.re[source];
                packed[(width + bin) * spectrogram.frames + frame] = spectrogram.im[source];
            }
        }
        let normalized = global_norm(
            compute,
            &Sequences {
                data: packed,
                batch: 1,
                channels: 2 * width,
                length: spectrogram.frames,
            },
            &band_weights.norm,
            f32::EPSILON,
        )?;
        let projected = pointwise(compute, &normalized, &band_weights.projection)?;
        for channel in 0..channels {
            for frame in 0..spectrogram.frames {
                let output_index = grid_index(&features, 0, channel, band, frame);
                features.data[output_index] = projected.data[channel * spectrogram.frames + frame];
            }
        }
        first_bin += width;
    }

    let separated_features = recurrent(compute, &features, &weights.separator)?;
    let sources = variant.output_streams();
    let mut separated: Vec<Spectrogram> = (0..sources)
        .map(|_| Spectrogram {
            frames: spectrogram.frames,
            bins: spectrogram.bins,
            re: vec![0.0; spectrogram.frames * spectrogram.bins],
            im: vec![0.0; spectrogram.frames * spectrogram.bins],
        })
        .collect();

    first_bin = 0;
    for (band, ((&width, mask), _input_weights)) in widths
        .iter()
        .zip(&weights.masks)
        .zip(&weights.bands)
        .enumerate()
    {
        let mut band_features = Sequences {
            data: vec![0.0; channels * spectrogram.frames],
            batch: 1,
            channels,
            length: spectrogram.frames,
        };
        for channel in 0..channels {
            for frame in 0..spectrogram.frames {
                band_features.data[channel * spectrogram.frames + frame] = separated_features.data
                    [grid_index(&separated_features, 0, channel, band, frame)];
            }
        }
        prelu_in_place(&mut band_features.data, mask.slope);
        let mask_output = grouped_conv(compute, &band_features, &mask.projection)?;
        for bin in 0..width {
            for frame in 0..spectrogram.frames {
                let input_index = frame * spectrogram.bins + first_bin + bin;
                let input_real = spectrogram.re[input_index];
                let input_imag = spectrogram.im[input_index];
                let mut real_masks = vec![0.0f32; sources];
                let mut imag_masks = vec![0.0f32; sources];
                for source in 0..sources {
                    let amplitude_real =
                        mask_value(&mask_output, 0, 0, source, bin, frame, sources, width);
                    let amplitude_imag =
                        mask_value(&mask_output, 0, 1, source, bin, frame, sources, width);
                    let gate_real =
                        mask_value(&mask_output, 1, 0, source, bin, frame, sources, width);
                    let gate_imag =
                        mask_value(&mask_output, 1, 1, source, bin, frame, sources, width);
                    real_masks[source] = amplitude_real * sigmoid(gate_real);
                    imag_masks[source] = amplitude_imag * sigmoid(gate_imag);
                }
                let real_correction = (real_masks.iter().sum::<f32>() - 1.0) / sources as f32;
                let imag_correction = imag_masks.iter().sum::<f32>() / sources as f32;
                for source in 0..sources {
                    let real_mask = real_masks[source] - real_correction;
                    let imag_mask = imag_masks[source] - imag_correction;
                    separated[source].re[input_index] =
                        input_real * real_mask - input_imag * imag_mask;
                    separated[source].im[input_index] =
                        input_real * imag_mask + input_imag * real_mask;
                }
            }
        }
        first_bin += width;
    }

    let mut attrs = IstftAttrs::new(n_fft, hop);
    attrs.length = Some(pcm.len());
    separated
        .iter()
        .map(|spectrum| istft(spectrum, &attrs))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn mask_value(
    output: &Sequences,
    amplitude_or_gate: usize,
    real_or_imag: usize,
    source: usize,
    bin: usize,
    frame: usize,
    sources: usize,
    width: usize,
) -> f32 {
    let channel = (((amplitude_or_gate * 2 + real_or_imag) * sources + source) * width) + bin;
    output.data[channel * output.length + frame]
}

fn recurrent(compute: &Compute, input: &Grid, weights: &super::weights::Separator) -> Result<Grid> {
    let mixture = input.clone();
    let mut hidden = input.clone();
    for iteration in 0..ITERATIONS {
        if iteration > 0 {
            add_in_place(&mut hidden.data, &mixture.data)?;
            let flat = Sequences {
                data: hidden.data,
                batch: hidden.batch,
                channels: hidden.channels,
                length: hidden.rows * hidden.columns,
            };
            let mut mixed = grouped_conv(compute, &flat, &weights.concat)?;
            prelu_in_place(&mut mixed.data, weights.concat_slope);
            hidden = Grid {
                data: mixed.data,
                batch: mixture.batch,
                channels: mixture.channels,
                rows: mixture.rows,
                columns: mixture.columns,
            };
        }
        hidden = frequency_frame_process(compute, &hidden, weights)?;
    }
    Ok(hidden)
}

fn frequency_frame_process(
    compute: &Compute,
    input: &Grid,
    weights: &super::weights::Separator,
) -> Result<Grid> {
    let frequency_input = grid_to_frequency_sequences(input);
    let frequency_hidden = uconv(compute, &frequency_input, &weights.frequency.uconv)?;
    let mut frequency_grid =
        frequency_sequences_to_grid(&frequency_hidden, input.batch, input.rows, input.columns)?;
    frequency_grid = attention_dim4(compute, &frequency_grid, &weights.frequency.attention)?;
    frequency_grid = layer_norm_grid(compute, &frequency_grid, &weights.frequency.norm)?;
    frequency_grid = swap_grid_axes(&frequency_grid);
    add_in_place(&mut frequency_grid.data, &input.data)?;

    let frame_input = grid_to_frame_sequences(&frequency_grid);
    let frame_hidden = uconv(compute, &frame_input, &weights.frame.uconv)?;
    let mut frame_grid =
        frame_sequences_to_grid(&frame_hidden, input.batch, input.rows, input.columns)?;
    frame_grid = attention_dim4(compute, &frame_grid, &weights.frame.attention)?;
    frame_grid = layer_norm_grid(compute, &frame_grid, &weights.frame.norm)?;
    add_in_place(&mut frame_grid.data, &frequency_grid.data)?;
    Ok(frame_grid)
}

fn uconv(compute: &Compute, input: &Sequences, weights: &UConv) -> Result<Sequences> {
    let residual = input.clone();
    let mut projected = pointwise(compute, input, &weights.projection)?;
    projected = global_norm(
        compute,
        &projected,
        &weights.projection_norm,
        GLOBAL_NORM_EPS,
    )?;
    prelu_in_place(&mut projected.data, weights.projection_slope);

    let mut scales = Vec::with_capacity(weights.downsample.len());
    let mut hidden = projected;
    for stage in &weights.downsample {
        hidden = depth_norm(compute, &hidden, stage)?;
        scales.push(hidden.clone());
    }
    let deepest_length = scales
        .last()
        .map(|value| value.length)
        .ok_or_else(|| VokraError::ModelLoad("tiger: UConv has no scales".to_owned()))?;
    let mut global = Sequences {
        data: vec![0.0; input.batch * super::INTERNAL_CHANNELS * deepest_length],
        batch: input.batch,
        channels: super::INTERNAL_CHANNELS,
        length: deepest_length,
    };
    for scale in &scales {
        let pooled = adaptive_avg_pool(scale, deepest_length)?;
        add_in_place(&mut global.data, &pooled.data)?;
    }
    global = mlp(compute, &global, &weights.global_mlp)?;

    let mut fused = Vec::with_capacity(scales.len());
    for (scale, injection_weights) in scales.iter().zip(&weights.scale_fusion) {
        fused.push(injection(compute, scale, &global, injection_weights)?);
    }
    if fused.len() != 5 || weights.expansion.len() != 4 {
        return Err(VokraError::ModelLoad(
            "tiger: UConv depth must be exactly five".to_owned(),
        ));
    }
    // Preserve the official source's unusual first expansion exactly:
    // last_layer[3](x_fused[3], x_fused[2]); x_fused[4] is not consumed.
    let mut expanded = injection(compute, &fused[3], &fused[2], &weights.expansion[3])?;
    for stage in (0..3).rev() {
        expanded = injection(compute, &fused[stage], &expanded, &weights.expansion[stage])?;
    }
    let mut output = pointwise(compute, &expanded, &weights.residual_projection)?;
    add_in_place(&mut output.data, &residual.data)?;
    Ok(output)
}

fn mlp(compute: &Compute, input: &Sequences, weights: &Mlp) -> Result<Sequences> {
    let mut hidden = pointwise_norm(compute, input, &weights.first)?;
    hidden = grouped_conv(compute, &hidden, &weights.depthwise)?;
    for value in &mut hidden.data {
        *value = value.max(0.0);
    }
    pointwise_norm(compute, &hidden, &weights.second)
}

fn pointwise_norm(
    compute: &Compute,
    input: &Sequences,
    weights: &PointwiseNorm,
) -> Result<Sequences> {
    let projected = pointwise(compute, input, &weights.conv)?;
    global_norm(compute, &projected, &weights.norm, GLOBAL_NORM_EPS)
}

fn depth_norm(compute: &Compute, input: &Sequences, weights: &DepthNorm) -> Result<Sequences> {
    let convolved = grouped_conv(compute, input, &weights.conv)?;
    global_norm(compute, &convolved, &weights.norm, GLOBAL_NORM_EPS)
}

fn injection(
    compute: &Compute,
    local: &Sequences,
    global: &Sequences,
    weights: &Injection,
) -> Result<Sequences> {
    if local.batch != global.batch || local.channels != global.channels {
        return Err(VokraError::InvalidArgument(
            "tiger: injection batch/channel mismatch".to_owned(),
        ));
    }
    let local_hidden = depth_norm(compute, local, &weights.local)?;
    let mut gate = depth_norm(compute, global, &weights.gate)?;
    for value in &mut gate.data {
        *value = sigmoid(*value);
    }
    let gate = nearest_interpolate(&gate, local.length)?;
    let global_hidden = depth_norm(compute, global, &weights.global)?;
    let global_hidden = nearest_interpolate(&global_hidden, local.length)?;
    let mut output = local_hidden;
    for ((value, gate), global) in output
        .data
        .iter_mut()
        .zip(gate.data)
        .zip(global_hidden.data)
    {
        *value = *value * gate + global;
    }
    Ok(output)
}

fn attention_dim4(compute: &Compute, input: &Grid, weights: &Attention) -> Result<Grid> {
    let transposed = swap_grid_axes(input);
    let residual = transposed.clone();
    if weights.queries.len() != 4 || weights.keys.len() != 4 || weights.values.len() != 4 {
        return Err(VokraError::ModelLoad(
            "tiger: attention must contain four heads".to_owned(),
        ));
    }
    let value_channels = input.channels / 4;
    let mut combined = Grid {
        data: vec![0.0; transposed.batch * input.channels * transposed.rows * transposed.columns],
        batch: transposed.batch,
        channels: input.channels,
        rows: transposed.rows,
        columns: transposed.columns,
    };
    for head in 0..4 {
        let query = attention_projection(compute, &transposed, &weights.queries[head])?;
        let key = attention_projection(compute, &transposed, &weights.keys[head])?;
        let value = attention_projection(compute, &transposed, &weights.values[head])?;
        let embedding = query.channels * query.columns;
        let value_embedding = value_channels * value.columns;
        let scale = (embedding as f32).sqrt().recip();
        for batch in 0..transposed.batch {
            let mut q_rows = vec![0.0f32; transposed.rows * embedding];
            let mut k_transposed = vec![0.0f32; embedding * transposed.rows];
            let mut v_rows = vec![0.0f32; transposed.rows * value_embedding];
            for row in 0..transposed.rows {
                for channel in 0..query.channels {
                    for column in 0..query.columns {
                        let inner = channel * query.columns + column;
                        q_rows[row * embedding + inner] =
                            query.data[grid_index(&query, batch, channel, row, column)];
                        k_transposed[inner * transposed.rows + row] =
                            key.data[grid_index(&key, batch, channel, row, column)];
                    }
                }
                for channel in 0..value_channels {
                    for column in 0..value.columns {
                        v_rows[row * value_embedding + channel * value.columns + column] =
                            value.data[grid_index(&value, batch, channel, row, column)];
                    }
                }
            }
            let mut logits = vec![0.0f32; transposed.rows * transposed.rows];
            compute.gemm_f32(
                transposed.rows,
                transposed.rows,
                embedding,
                &q_rows,
                &k_transposed,
                None,
                &mut logits,
            )?;
            for value in &mut logits {
                *value *= scale;
            }
            let mut probabilities = vec![0.0f32; logits.len()];
            compute.softmax_f32(
                &logits,
                &mut probabilities,
                transposed.rows,
                transposed.rows,
            )?;
            let mut context = vec![0.0f32; transposed.rows * value_embedding];
            compute.gemm_f32(
                transposed.rows,
                value_embedding,
                transposed.rows,
                &probabilities,
                &v_rows,
                None,
                &mut context,
            )?;
            for row in 0..transposed.rows {
                for channel in 0..value_channels {
                    for column in 0..transposed.columns {
                        let output_channel = head * value_channels + channel;
                        let source = row * value_embedding + channel * transposed.columns + column;
                        let target = grid_index(&combined, batch, output_channel, row, column);
                        combined.data[target] = context[source];
                    }
                }
            }
        }
    }
    let mut output = attention_projection(compute, &combined, &weights.output)?;
    add_in_place(&mut output.data, &residual.data)?;
    Ok(swap_grid_axes(&output))
}

fn attention_projection(
    compute: &Compute,
    input: &Grid,
    weights: &AttentionProjection,
) -> Result<Grid> {
    let sequence = Sequences {
        data: input.data.clone(),
        batch: input.batch,
        channels: input.channels,
        length: input.rows * input.columns,
    };
    let mut output = pointwise(compute, &sequence, &weights.projection)?;
    prelu_in_place(&mut output.data, weights.slope);
    layer_norm_grid(
        compute,
        &Grid {
            data: output.data,
            batch: input.batch,
            channels: weights.projection.output,
            rows: input.rows,
            columns: input.columns,
        },
        &weights.norm,
    )
}

fn pointwise(compute: &Compute, input: &Sequences, weights: &Pointwise) -> Result<Sequences> {
    if input.channels != weights.input
        || weights.weight_t.len() != weights.input * weights.output
        || weights
            .bias
            .as_ref()
            .is_some_and(|bias| bias.len() != weights.output)
    {
        return Err(VokraError::InvalidArgument(
            "tiger: pointwise projection shape mismatch".to_owned(),
        ));
    }
    let rows = input.batch * input.length;
    let mut row_major = vec![0.0f32; rows * input.channels];
    for batch in 0..input.batch {
        for position in 0..input.length {
            for channel in 0..input.channels {
                row_major[(batch * input.length + position) * input.channels + channel] =
                    input.data[(batch * input.channels + channel) * input.length + position];
            }
        }
    }
    let mut projected = vec![0.0f32; rows * weights.output];
    compute.gemm_f32(
        rows,
        weights.output,
        weights.input,
        &row_major,
        &weights.weight_t,
        weights.bias.as_deref(),
        &mut projected,
    )?;
    let mut output = Sequences {
        data: vec![0.0; input.batch * weights.output * input.length],
        batch: input.batch,
        channels: weights.output,
        length: input.length,
    };
    for batch in 0..input.batch {
        for position in 0..input.length {
            for channel in 0..weights.output {
                output.data[(batch * weights.output + channel) * input.length + position] =
                    projected[(batch * input.length + position) * weights.output + channel];
            }
        }
    }
    Ok(output)
}

fn grouped_conv(compute: &Compute, input: &Sequences, weights: &GroupedConv) -> Result<Sequences> {
    if input.channels != weights.input
        || weights.groups == 0
        || weights.input % weights.groups != 0
        || weights.output % weights.groups != 0
    {
        return Err(VokraError::InvalidArgument(
            "tiger: grouped convolution shape mismatch".to_owned(),
        ));
    }
    let padded = input
        .length
        .checked_add(2 * weights.padding)
        .ok_or_else(|| {
            VokraError::InvalidArgument("tiger: convolution length overflow".to_owned())
        })?;
    if padded < weights.kernel || weights.stride == 0 {
        return Err(VokraError::InvalidArgument(
            "tiger: convolution kernel exceeds padded input".to_owned(),
        ));
    }
    let output_length = (padded - weights.kernel) / weights.stride + 1;
    let mut repeated_weight = Vec::with_capacity(input.batch * weights.weight.len());
    let mut repeated_bias = weights
        .bias
        .as_ref()
        .map(|bias| Vec::with_capacity(input.batch * bias.len()));
    for _ in 0..input.batch {
        repeated_weight.extend_from_slice(&weights.weight);
        if let (Some(output), Some(bias)) = (&mut repeated_bias, &weights.bias) {
            output.extend_from_slice(bias);
        }
    }
    let mut output = Sequences {
        data: vec![0.0; input.batch * weights.output * output_length],
        batch: input.batch,
        channels: weights.output,
        length: output_length,
    };
    compute.grouped_conv1d_f32(
        &input.data,
        input.batch * weights.input,
        input.length,
        &repeated_weight,
        input.batch * weights.output,
        weights.kernel,
        repeated_bias.as_deref(),
        weights.stride,
        weights.padding,
        input.batch * weights.groups,
        &mut output.data,
    )?;
    Ok(output)
}

fn global_norm(
    compute: &Compute,
    input: &Sequences,
    weights: &Norm,
    epsilon: f32,
) -> Result<Sequences> {
    if weights.gamma.len() != input.channels || weights.beta.len() != input.channels {
        return Err(VokraError::ModelLoad(
            "tiger: global normalization affine shape mismatch".to_owned(),
        ));
    }
    let columns = input.channels * input.length;
    let mut gamma = Vec::with_capacity(columns);
    let mut beta = Vec::with_capacity(columns);
    for channel in 0..input.channels {
        gamma.extend(std::iter::repeat_n(weights.gamma[channel], input.length));
        beta.extend(std::iter::repeat_n(weights.beta[channel], input.length));
    }
    let mut output = input.clone();
    compute.layer_norm_f32(
        &input.data,
        &mut output.data,
        input.batch,
        columns,
        &gamma,
        &beta,
        epsilon,
    )?;
    Ok(output)
}

fn layer_norm_grid(compute: &Compute, input: &Grid, weights: &Norm) -> Result<Grid> {
    if weights.gamma.len() != input.channels || weights.beta.len() != input.channels {
        return Err(VokraError::ModelLoad(
            "tiger: LayerNormalization4D affine shape mismatch".to_owned(),
        ));
    }
    let positions = input.batch * input.rows * input.columns;
    let mut row_major = vec![0.0f32; positions * input.channels];
    for batch in 0..input.batch {
        for row in 0..input.rows {
            for column in 0..input.columns {
                let position = (batch * input.rows + row) * input.columns + column;
                for channel in 0..input.channels {
                    row_major[position * input.channels + channel] =
                        input.data[grid_index(input, batch, channel, row, column)];
                }
            }
        }
    }
    let mut normalized = vec![0.0f32; row_major.len()];
    compute.layer_norm_f32(
        &row_major,
        &mut normalized,
        positions,
        input.channels,
        &weights.gamma,
        &weights.beta,
        LAYER_NORM_EPS,
    )?;
    let mut output = input.clone();
    for batch in 0..input.batch {
        for row in 0..input.rows {
            for column in 0..input.columns {
                let position = (batch * input.rows + row) * input.columns + column;
                for channel in 0..input.channels {
                    let target = grid_index(&output, batch, channel, row, column);
                    output.data[target] = normalized[position * input.channels + channel];
                }
            }
        }
    }
    Ok(output)
}

fn adaptive_avg_pool(input: &Sequences, output_length: usize) -> Result<Sequences> {
    if output_length == 0 || input.length == 0 {
        return Err(VokraError::InvalidArgument(
            "tiger: adaptive average pool lengths must be positive".to_owned(),
        ));
    }
    let mut output = Sequences {
        data: vec![0.0; input.batch * input.channels * output_length],
        batch: input.batch,
        channels: input.channels,
        length: output_length,
    };
    for batch in 0..input.batch {
        for channel in 0..input.channels {
            for target in 0..output_length {
                let start = target * input.length / output_length;
                let end = ((target + 1) * input.length).div_ceil(output_length);
                let source_base = (batch * input.channels + channel) * input.length;
                let sum: f32 = input.data[source_base + start..source_base + end]
                    .iter()
                    .sum();
                output.data[(batch * input.channels + channel) * output_length + target] =
                    sum / (end - start) as f32;
            }
        }
    }
    Ok(output)
}

fn nearest_interpolate(input: &Sequences, output_length: usize) -> Result<Sequences> {
    if input.length == 0 || output_length == 0 {
        return Err(VokraError::InvalidArgument(
            "tiger: nearest interpolation lengths must be positive".to_owned(),
        ));
    }
    let mut output = Sequences {
        data: vec![0.0; input.batch * input.channels * output_length],
        batch: input.batch,
        channels: input.channels,
        length: output_length,
    };
    for batch in 0..input.batch {
        for channel in 0..input.channels {
            for target in 0..output_length {
                let source = (target * input.length / output_length).min(input.length - 1);
                output.data[(batch * input.channels + channel) * output_length + target] =
                    input.data[(batch * input.channels + channel) * input.length + source];
            }
        }
    }
    Ok(output)
}

fn grid_to_frequency_sequences(input: &Grid) -> Sequences {
    let mut output = Sequences {
        data: vec![0.0; input.batch * input.columns * input.channels * input.rows],
        batch: input.batch * input.columns,
        channels: input.channels,
        length: input.rows,
    };
    for batch in 0..input.batch {
        for column in 0..input.columns {
            let sequence = batch * input.columns + column;
            for channel in 0..input.channels {
                for row in 0..input.rows {
                    output.data[(sequence * input.channels + channel) * input.rows + row] =
                        input.data[grid_index(input, batch, channel, row, column)];
                }
            }
        }
    }
    output
}

fn frequency_sequences_to_grid(
    input: &Sequences,
    batch: usize,
    bands: usize,
    frames: usize,
) -> Result<Grid> {
    if input.batch != batch * frames || input.length != bands {
        return Err(VokraError::InvalidArgument(
            "tiger: frequency sequence reshape mismatch".to_owned(),
        ));
    }
    let mut output = Grid {
        data: vec![0.0; batch * input.channels * frames * bands],
        batch,
        channels: input.channels,
        rows: frames,
        columns: bands,
    };
    for item in 0..batch {
        for frame in 0..frames {
            let sequence = item * frames + frame;
            for channel in 0..input.channels {
                for band in 0..bands {
                    let target = grid_index(&output, item, channel, frame, band);
                    output.data[target] =
                        input.data[(sequence * input.channels + channel) * bands + band];
                }
            }
        }
    }
    Ok(output)
}

fn grid_to_frame_sequences(input: &Grid) -> Sequences {
    let mut output = Sequences {
        data: vec![0.0; input.batch * input.rows * input.channels * input.columns],
        batch: input.batch * input.rows,
        channels: input.channels,
        length: input.columns,
    };
    for batch in 0..input.batch {
        for row in 0..input.rows {
            let sequence = batch * input.rows + row;
            for channel in 0..input.channels {
                for column in 0..input.columns {
                    output.data[(sequence * input.channels + channel) * input.columns + column] =
                        input.data[grid_index(input, batch, channel, row, column)];
                }
            }
        }
    }
    output
}

fn frame_sequences_to_grid(
    input: &Sequences,
    batch: usize,
    bands: usize,
    frames: usize,
) -> Result<Grid> {
    if input.batch != batch * bands || input.length != frames {
        return Err(VokraError::InvalidArgument(
            "tiger: frame sequence reshape mismatch".to_owned(),
        ));
    }
    let mut output = Grid {
        data: vec![0.0; batch * input.channels * bands * frames],
        batch,
        channels: input.channels,
        rows: bands,
        columns: frames,
    };
    for item in 0..batch {
        for band in 0..bands {
            let sequence = item * bands + band;
            for channel in 0..input.channels {
                for frame in 0..frames {
                    let target = grid_index(&output, item, channel, band, frame);
                    output.data[target] =
                        input.data[(sequence * input.channels + channel) * frames + frame];
                }
            }
        }
    }
    Ok(output)
}

fn swap_grid_axes(input: &Grid) -> Grid {
    let mut output = Grid {
        data: vec![0.0; input.data.len()],
        batch: input.batch,
        channels: input.channels,
        rows: input.columns,
        columns: input.rows,
    };
    for batch in 0..input.batch {
        for channel in 0..input.channels {
            for row in 0..input.rows {
                for column in 0..input.columns {
                    let target = grid_index(&output, batch, channel, column, row);
                    output.data[target] =
                        input.data[grid_index(input, batch, channel, row, column)];
                }
            }
        }
    }
    output
}

fn grid_index(grid: &Grid, batch: usize, channel: usize, row: usize, column: usize) -> usize {
    (((batch * grid.channels + channel) * grid.rows + row) * grid.columns) + column
}

fn prelu_in_place(values: &mut [f32], slope: f32) {
    for value in values {
        if *value < 0.0 {
            *value *= slope;
        }
    }
}

fn add_in_place(output: &mut [f32], residual: &[f32]) -> Result<()> {
    if output.len() != residual.len() {
        return Err(VokraError::InvalidArgument(format!(
            "tiger: residual lengths differ ({} vs {})",
            output.len(),
            residual.len()
        )));
    }
    for (output, residual) in output.iter_mut().zip(residual) {
        *output += residual;
    }
    Ok(())
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_swaps_and_sequence_views_roundtrip() {
        let grid = Grid {
            data: (0..2 * 3 * 4).map(|value| value as f32).collect(),
            batch: 1,
            channels: 2,
            rows: 3,
            columns: 4,
        };
        assert_eq!(swap_grid_axes(&swap_grid_axes(&grid)).data, grid.data);
        let frequency = grid_to_frequency_sequences(&grid);
        let restored = frequency_sequences_to_grid(&frequency, 1, 3, 4)
            .map(|value| swap_grid_axes(&value))
            .unwrap();
        assert_eq!(restored.data, grid.data);
        let frames = grid_to_frame_sequences(&grid);
        let restored = frame_sequences_to_grid(&frames, 1, 3, 4).unwrap();
        assert_eq!(restored.data, grid.data);
    }

    #[test]
    fn nearest_and_adaptive_pool_match_pytorch_index_rules() {
        let input = Sequences {
            data: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            batch: 1,
            channels: 1,
            length: 5,
        };
        assert_eq!(
            nearest_interpolate(&input, 3).unwrap().data,
            vec![0.0, 1.0, 3.0]
        );
        assert_eq!(adaptive_avg_pool(&input, 2).unwrap().data, vec![1.0, 3.5]);
    }

    #[test]
    fn stable_sigmoid_handles_extremes() {
        assert_eq!(sigmoid(1_000.0), 1.0);
        assert_eq!(sigmoid(-1_000.0), 0.0);
        assert!((sigmoid(0.0) - 0.5).abs() <= f32::EPSILON);
    }
}
