use vokra_core::{Result, VokraError};
use vokra_ops::fft::RealFftPlan;

use crate::compute::Compute;

use super::weights::{
    AliasFreeActivation, Conv1d, ConvTranspose1d, DecoderWeights, EncoderWeights, FacodecWeights,
    Linear, QuantizerLayer, ResidualUnit, TRANSFORMER_FFN, TRANSFORMER_HEADS, TransformerWeights,
};
use super::{CODEBOOK_DIM, CODEBOOK_SIZE, DIM, FacodecEncoded, LABEL, NUM_CODEBOOKS};

const LAYER_NORM_EPS: f32 = 1.0e-5;
const ALIAS_RATIO: usize = 2;
const MEL_N_FFT: usize = 1_024;
const MEL_HOP: usize = 200;
const MEL_WINDOW: usize = 800;
const MEL_BINS: usize = 80;
const PROSODY_BINS: usize = 20;
const REFLECT_PAD: usize = (MEL_N_FFT - MEL_HOP) / 2;

pub(super) fn encode(
    pcm: &[f32],
    weights: &FacodecWeights,
    compute: &Compute,
) -> Result<FacodecEncoded> {
    if pcm.len() <= REFLECT_PAD {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: PCM must contain more than {REFLECT_PAD} samples for the official reflect-padded mel frontend, got {}",
            pcm.len()
        )));
    }
    reject_non_finite("PCM", pcm)?;

    let (latent_channel_major, frames) = encoder_forward(pcm, &weights.encoder, compute)?;
    let latent = channel_to_frame(&latent_channel_major, DIM, frames);
    let prosody = prosody_features(pcm, &weights.encoder)?;
    if prosody.len() != frames * PROSODY_BINS {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: encoder produced {frames} frames but mel frontend produced {}; the input length does not satisfy the official V2 frame contract",
            prosody.len() / PROSODY_BINS
        )));
    }

    let prosody = linear_forward(&prosody, frames, &weights.decoder.melspec_linear, compute)?;
    let prosody = transformer_forward(&prosody, frames, &weights.decoder.melspec_encoder, compute)?;

    let (prosody_codes, prosody_quantized) = quantize_group(
        &prosody,
        frames,
        &weights.decoder.quantizer.groups[0],
        compute,
    )?;
    let (content_codes, content_quantized) = quantize_group(
        &latent,
        frames,
        &weights.decoder.quantizer.groups[1],
        compute,
    )?;
    let mut residual_input = latent.clone();
    for ((value, prosody), content) in residual_input
        .iter_mut()
        .zip(&prosody_quantized)
        .zip(&content_quantized)
    {
        *value -= prosody + content;
    }
    let (detail_codes, _) = quantize_group(
        &residual_input,
        frames,
        &weights.decoder.quantizer.groups[2],
        compute,
    )?;

    let timbre = transformer_forward(&latent, frames, &weights.decoder.timbre_encoder, compute)?;
    let mut speaker_embedding = vec![0.0f32; DIM];
    for row in timbre.chunks_exact(DIM) {
        for (sum, value) in speaker_embedding.iter_mut().zip(row) {
            *sum += *value;
        }
    }
    for value in &mut speaker_embedding {
        *value /= frames as f32;
    }

    let mut codes = vec![0u32; frames * NUM_CODEBOOKS];
    for frame in 0..frames {
        codes[frame * NUM_CODEBOOKS] = prosody_codes[frame];
        codes[frame * NUM_CODEBOOKS + 1] = content_codes[frame * 2];
        codes[frame * NUM_CODEBOOKS + 2] = content_codes[frame * 2 + 1];
        codes[frame * NUM_CODEBOOKS + 3] = detail_codes[frame * 3];
        codes[frame * NUM_CODEBOOKS + 4] = detail_codes[frame * 3 + 1];
        codes[frame * NUM_CODEBOOKS + 5] = detail_codes[frame * 3 + 2];
    }
    reject_non_finite("speaker embedding", &speaker_embedding)?;
    Ok(FacodecEncoded {
        frames,
        codes,
        speaker_embedding,
        input_samples: pcm.len(),
    })
}

pub(super) fn decode(
    encoded: &FacodecEncoded,
    weights: &DecoderWeights,
    compute: &Compute,
) -> Result<Vec<f32>> {
    encoded.validate()?;
    let features = compute.dac_rvq_f32(
        &encoded.codes,
        encoded.frames,
        &weights.quantizer.codebook_tables,
        &weights.quantizer.output_projections,
        &weights.quantizer.attrs,
    )?;
    decoder_forward(
        &features,
        encoded.frames,
        &encoded.speaker_embedding,
        weights,
        compute,
    )
}

fn encoder_forward(
    pcm: &[f32],
    weights: &EncoderWeights,
    compute: &Compute,
) -> Result<(Vec<f32>, usize)> {
    let (mut hidden, mut time) = conv1d_forward(pcm, pcm.len(), &weights.pre, compute)?;
    for stage in &weights.stages {
        for residual in &stage.residuals {
            hidden = residual_forward(&hidden, time, residual, compute)?;
        }
        hidden = alias_free_activation(
            &hidden,
            stage.downsample.input,
            time,
            &stage.activation,
            compute,
        )?;
        let (next, next_time) = conv1d_forward(&hidden, time, &stage.downsample, compute)?;
        hidden = next;
        time = next_time;
    }
    hidden = alias_free_activation(&hidden, 512, time, &weights.post_activation, compute)?;
    let (hidden, final_time) = conv1d_forward(&hidden, time, &weights.post, compute)?;
    reject_non_finite("encoder output", &hidden)?;
    Ok((hidden, final_time))
}

fn decoder_forward(
    features: &[f32],
    frames: usize,
    speaker_embedding: &[f32],
    weights: &DecoderWeights,
    compute: &Compute,
) -> Result<Vec<f32>> {
    if features.len() != frames * DIM || speaker_embedding.len() != DIM {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: decoder feature/speaker shape mismatch: features={} for {frames}x{DIM}, speaker={} expected {DIM}",
            features.len(),
            speaker_embedding.len()
        )));
    }
    let style = linear_forward(speaker_embedding, 1, &weights.timbre_linear, compute)?;
    let (gamma, beta) = style.split_at(DIM);
    let ones = vec![1.0f32; DIM];
    let zeros = vec![0.0f32; DIM];
    let mut normalized = vec![0.0f32; features.len()];
    compute.layer_norm_f32(
        features,
        &mut normalized,
        frames,
        DIM,
        &ones,
        &zeros,
        LAYER_NORM_EPS,
    )?;
    for row in normalized.chunks_exact_mut(DIM) {
        for channel in 0..DIM {
            row[channel] = row[channel] * gamma[channel] + beta[channel];
        }
    }

    let mut time = frames;
    let channel_major = frame_to_channel(&normalized, frames, DIM);
    let (mut hidden, pre_time) = conv1d_forward(&channel_major, time, &weights.pre, compute)?;
    time = pre_time;
    for stage in &weights.stages {
        hidden = alias_free_activation(
            &hidden,
            stage.upsample.input,
            time,
            &stage.activation,
            compute,
        )?;
        let (next, next_time) = conv_transpose1d_forward(&hidden, time, &stage.upsample, compute)?;
        hidden = next;
        time = next_time;
        for residual in &stage.residuals {
            hidden = residual_forward(&hidden, time, residual, compute)?;
        }
    }
    hidden = alias_free_activation(&hidden, 64, time, &weights.post_activation, compute)?;
    let (pcm, pcm_time) = conv1d_forward(&hidden, time, &weights.post, compute)?;
    if pcm_time != frames * MEL_HOP {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: decoder produced {pcm_time} samples for {frames} frames, expected {}",
            frames * MEL_HOP
        )));
    }
    let mut output = vec![0.0f32; pcm.len()];
    compute.tanh_f32(&pcm, &mut output)?;
    reject_non_finite("decoder PCM", &output)?;
    Ok(output)
}

fn quantize_group(
    input: &[f32],
    frames: usize,
    layers: &[QuantizerLayer],
    compute: &Compute,
) -> Result<(Vec<u32>, Vec<f32>)> {
    if input.len() != frames * DIM || frames == 0 || layers.is_empty() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: quantizer input has {} values for {frames}x{DIM} and {} layers",
            input.len(),
            layers.len()
        )));
    }
    let mut residual = input.to_vec();
    let mut summed = vec![0.0f32; input.len()];
    let mut codes = vec![0u32; frames * layers.len()];
    for (layer_index, layer) in layers.iter().enumerate() {
        let projected = linear_forward(&residual, frames, &layer.input_projection, compute)?;
        let mut normalized = projected.clone();
        for row in normalized.chunks_exact_mut(CODEBOOK_DIM) {
            let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
            if !norm.is_finite() {
                return Err(VokraError::InvalidArgument(format!(
                    "{LABEL}: quantizer projected frame has invalid L2 norm {norm}"
                )));
            }
            let denominator = norm.max(1.0e-12);
            for value in row {
                *value /= denominator;
            }
        }
        let mut scores = vec![0.0f32; frames * CODEBOOK_SIZE];
        compute.gemm_f32(
            frames,
            CODEBOOK_SIZE,
            CODEBOOK_DIM,
            &normalized,
            &layer.normalized_codebook_t,
            None,
            &mut scores,
        )?;
        let mut low = vec![0.0f32; frames * CODEBOOK_DIM];
        for frame in 0..frames {
            let row = &scores[frame * CODEBOOK_SIZE..(frame + 1) * CODEBOOK_SIZE];
            let mut index = 0usize;
            // Official FVQ maximizes negative squared Euclidean distance
            // after F.normalize. The encoding norm is constant across the
            // row and can be omitted, but the codebook norm must remain so a
            // legal all-zero row follows PyTorch's eps behavior exactly.
            let mut best = 2.0 * row[0] - layer.normalized_codebook_norm2[0];
            if !best.is_finite() {
                return Err(VokraError::InvalidArgument(format!(
                    "{LABEL}: non-finite codebook score at frame {frame}, code 0"
                )));
            }
            for (candidate, &dot) in row.iter().enumerate().skip(1) {
                let score = 2.0 * dot - layer.normalized_codebook_norm2[candidate];
                if !score.is_finite() {
                    return Err(VokraError::InvalidArgument(format!(
                        "{LABEL}: non-finite codebook score at frame {frame}, code {candidate}"
                    )));
                }
                // torch.max returns the first index on a tie.
                if score > best {
                    best = score;
                    index = candidate;
                }
            }
            codes[frame * layers.len() + layer_index] = index as u32;
            low[frame * CODEBOOK_DIM..(frame + 1) * CODEBOOK_DIM]
                .copy_from_slice(&layer.codebook[index * CODEBOOK_DIM..(index + 1) * CODEBOOK_DIM]);
        }
        let quantized = linear_forward(&low, frames, &layer.output_projection, compute)?;
        for ((residual, sum), value) in residual.iter_mut().zip(&mut summed).zip(quantized) {
            *residual -= value;
            *sum += value;
        }
    }
    Ok((codes, summed))
}

fn transformer_forward(
    input: &[f32],
    frames: usize,
    weights: &TransformerWeights,
    compute: &Compute,
) -> Result<Vec<f32>> {
    if frames == 0 || input.len() != frames * DIM {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: transformer input has {} values for {frames}x{DIM}",
            input.len()
        )));
    }
    let mut hidden = input.to_vec();
    // The official Amphion PositionalEncoding is sequence-first, but its
    // TransformerEncoder passes batch-first `[B,T,D]`. For the released
    // batch-size-one inference route `x.size(0) == 1`, so `pe[:1]` broadcasts
    // row zero across every frame. Reproducing the intended frame-indexed PE
    // would diverge from the actual official reference.
    for frame in 0..frames {
        for channel in 0..DIM {
            hidden[frame * DIM + channel] += weights.position[channel];
        }
    }
    for layer in &weights.layers {
        let mut normalized = vec![0.0f32; hidden.len()];
        compute.layer_norm_f32(
            &hidden,
            &mut normalized,
            frames,
            DIM,
            &layer.norm1_weight,
            &layer.norm1_bias,
            LAYER_NORM_EPS,
        )?;
        let qkv = linear_forward(&normalized, frames, &layer.attention_in, compute)?;
        let attended = multi_head_attention(&qkv, frames, compute)?;
        let projected = linear_forward(&attended, frames, &layer.attention_out, compute)?;
        for (value, residual) in hidden.iter_mut().zip(projected) {
            *value += residual;
        }

        compute.layer_norm_f32(
            &hidden,
            &mut normalized,
            frames,
            DIM,
            &layer.norm2_weight,
            &layer.norm2_bias,
            LAYER_NORM_EPS,
        )?;
        let channel_major = frame_to_channel(&normalized, frames, DIM);
        let mut conv = vec![0.0f32; frames * TRANSFORMER_FFN];
        compute.conv1d_f32(
            &channel_major,
            DIM,
            frames,
            &layer.ffn_conv_weight,
            TRANSFORMER_FFN,
            5,
            Some(&layer.ffn_conv_bias),
            1,
            2,
            &mut conv,
        )?;
        let mut activated = vec![0.0f32; conv.len()];
        compute.relu_f32(&conv, &mut activated)?;
        let activated = channel_to_frame(&activated, TRANSFORMER_FFN, frames);
        let projected = linear_forward(&activated, frames, &layer.ffn_out, compute)?;
        for (value, residual) in hidden.iter_mut().zip(projected) {
            *value += residual;
        }
    }
    let mut output = vec![0.0f32; hidden.len()];
    compute.layer_norm_f32(
        &hidden,
        &mut output,
        frames,
        DIM,
        &weights.last_norm_weight,
        &weights.last_norm_bias,
        LAYER_NORM_EPS,
    )?;
    Ok(output)
}

fn multi_head_attention(qkv: &[f32], frames: usize, compute: &Compute) -> Result<Vec<f32>> {
    let head_dim = DIM / TRANSFORMER_HEADS;
    if qkv.len() != frames * 3 * DIM {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: attention qkv has {} values, expected {}",
            qkv.len(),
            frames * 3 * DIM
        )));
    }
    let mut output = vec![0.0f32; frames * DIM];
    for head in 0..TRANSFORMER_HEADS {
        let mut query = vec![0.0f32; frames * head_dim];
        let mut key_t = vec![0.0f32; head_dim * frames];
        let mut value = vec![0.0f32; frames * head_dim];
        for frame in 0..frames {
            let source = frame * 3 * DIM;
            for channel in 0..head_dim {
                let model_channel = head * head_dim + channel;
                query[frame * head_dim + channel] = qkv[source + model_channel];
                key_t[channel * frames + frame] = qkv[source + DIM + model_channel];
                value[frame * head_dim + channel] = qkv[source + 2 * DIM + model_channel];
            }
        }
        let mut scores = vec![0.0f32; frames * frames];
        compute.gemm_f32(frames, frames, head_dim, &query, &key_t, None, &mut scores)?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        for score in &mut scores {
            *score *= scale;
        }
        let mut probabilities = vec![0.0f32; scores.len()];
        compute.softmax_f32(&scores, &mut probabilities, frames, frames)?;
        let mut head_output = vec![0.0f32; frames * head_dim];
        compute.gemm_f32(
            frames,
            head_dim,
            frames,
            &probabilities,
            &value,
            None,
            &mut head_output,
        )?;
        for frame in 0..frames {
            output[frame * DIM + head * head_dim..frame * DIM + (head + 1) * head_dim]
                .copy_from_slice(&head_output[frame * head_dim..(frame + 1) * head_dim]);
        }
    }
    Ok(output)
}

fn residual_forward(
    input: &[f32],
    time: usize,
    weights: &ResidualUnit,
    compute: &Compute,
) -> Result<Vec<f32>> {
    let channels = weights.first_conv.input;
    let activated =
        alias_free_activation(input, channels, time, &weights.first_activation, compute)?;
    let (hidden, first_time) = conv1d_forward(&activated, time, &weights.first_conv, compute)?;
    if first_time != time {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: residual dilated convolution changed time {time} -> {first_time}"
        )));
    }
    let activated =
        alias_free_activation(&hidden, channels, time, &weights.second_activation, compute)?;
    let (hidden, second_time) = conv1d_forward(&activated, time, &weights.second_conv, compute)?;
    if second_time != time || hidden.len() != input.len() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: residual 1x1 convolution changed shape {}x{time} -> {} values/{second_time}",
            channels,
            hidden.len()
        )));
    }
    Ok(input
        .iter()
        .zip(hidden)
        .map(|(left, right)| left + right)
        .collect())
}

fn alias_free_activation(
    input: &[f32],
    channels: usize,
    time: usize,
    weights: &AliasFreeActivation,
    compute: &Compute,
) -> Result<Vec<f32>> {
    if input.len() != channels * time
        || weights.upsample_filter.len() != 12
        || weights.downsample_filter.len() != 12
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: alias-free activation shape mismatch input={} channels={channels} time={time} up_filter={} down_filter={}",
            input.len(),
            weights.upsample_filter.len(),
            weights.downsample_filter.len()
        )));
    }
    // Official UpSample1d: replicate-pad five samples, depthwise
    // ConvTranspose1d(k=12,s=2), multiply by ratio, crop 15 on each side.
    let padded = replicate_pad_channel_major(input, channels, time, 5, 5)?;
    let padded_time = time + 10;
    let inserted_time = (padded_time - 1) * ALIAS_RATIO + 1;
    let mut inserted = vec![0.0f32; channels * inserted_time];
    for channel in 0..channels {
        for position in 0..padded_time {
            inserted[channel * inserted_time + position * ALIAS_RATIO] =
                padded[channel * padded_time + position];
        }
    }
    let reversed: Vec<f32> = weights
        .upsample_filter
        .iter()
        .rev()
        .map(|value| value * ALIAS_RATIO as f32)
        .collect();
    let up_weight = repeat_depthwise_kernel(&reversed, channels);
    let raw_time = inserted_time + reversed.len() - 1;
    let mut raw = vec![0.0f32; channels * raw_time];
    compute.grouped_conv1d_f32(
        &inserted,
        channels,
        inserted_time,
        &up_weight,
        channels,
        reversed.len(),
        None,
        1,
        reversed.len() - 1,
        channels,
        &mut raw,
    )?;
    let up_time = time * ALIAS_RATIO;
    let mut upsampled = vec![0.0f32; channels * up_time];
    for channel in 0..channels {
        let source = channel * raw_time + 15;
        upsampled[channel * up_time..(channel + 1) * up_time]
            .copy_from_slice(&raw[source..source + up_time]);
    }

    let alpha: Vec<f32> = weights.alpha.iter().map(|value| value.exp()).collect();
    let beta: Vec<f32> = weights.beta.iter().map(|value| value.exp()).collect();
    let mut activated = vec![0.0f32; upsampled.len()];
    compute.snake_beta_f32(&upsampled, &alpha, &beta, channels, up_time, &mut activated)?;

    // Official DownSample1d: replicate pad [5,6], depthwise k=12,s=2.
    let down_padded = replicate_pad_channel_major(&activated, channels, up_time, 5, 6)?;
    let down_padded_time = up_time + 11;
    let down_weight = repeat_depthwise_kernel(&weights.downsample_filter, channels);
    let mut output = vec![0.0f32; channels * time];
    compute.grouped_conv1d_f32(
        &down_padded,
        channels,
        down_padded_time,
        &down_weight,
        channels,
        weights.downsample_filter.len(),
        None,
        ALIAS_RATIO,
        0,
        channels,
        &mut output,
    )?;
    Ok(output)
}

fn conv1d_forward(
    input: &[f32],
    time: usize,
    weights: &Conv1d,
    compute: &Compute,
) -> Result<(Vec<f32>, usize)> {
    if input.len() != weights.input * time {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: conv1d input has {} values, expected {}x{time}",
            input.len(),
            weights.input
        )));
    }
    let effective_kernel = (weights.kernel - 1)
        .checked_mul(weights.dilation)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| VokraError::InvalidArgument(format!("{LABEL}: conv1d kernel overflow")))?;
    let padded = time
        .checked_add(2 * weights.padding)
        .ok_or_else(|| VokraError::InvalidArgument(format!("{LABEL}: conv1d time overflow")))?;
    if padded < effective_kernel || weights.stride == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: conv1d invalid time={time} kernel={} dilation={} padding={} stride={}",
            weights.kernel, weights.dilation, weights.padding, weights.stride
        )));
    }
    let output_time = (padded - effective_kernel) / weights.stride + 1;
    let expanded_weight;
    let weight = if weights.dilation == 1 {
        &weights.weight
    } else {
        let mut expanded = vec![0.0f32; weights.output * weights.input * effective_kernel];
        for output in 0..weights.output {
            for input_channel in 0..weights.input {
                for tap in 0..weights.kernel {
                    let source = (output * weights.input + input_channel) * weights.kernel + tap;
                    let destination = (output * weights.input + input_channel) * effective_kernel
                        + tap * weights.dilation;
                    expanded[destination] = weights.weight[source];
                }
            }
        }
        expanded_weight = expanded;
        &expanded_weight
    };
    let mut output = vec![0.0f32; weights.output * output_time];
    compute.conv1d_f32(
        input,
        weights.input,
        time,
        weight,
        weights.output,
        effective_kernel,
        Some(&weights.bias),
        weights.stride,
        weights.padding,
        &mut output,
    )?;
    Ok((output, output_time))
}

fn conv_transpose1d_forward(
    input: &[f32],
    time: usize,
    weights: &ConvTranspose1d,
    compute: &Compute,
) -> Result<(Vec<f32>, usize)> {
    if input.len() != weights.input * time || time == 0 || weights.stride == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: conv_transpose1d input has {} values for {}x{time}, stride={}",
            input.len(),
            weights.input,
            weights.stride
        )));
    }
    let output_time = (time - 1)
        .checked_mul(weights.stride)
        .and_then(|value| value.checked_add(weights.kernel + weights.output_padding))
        .and_then(|value| value.checked_sub(2 * weights.padding))
        .ok_or_else(|| {
            VokraError::InvalidArgument(format!("{LABEL}: conv_transpose1d output size overflow"))
        })?;
    let frame_major = channel_to_frame(input, weights.input, time);
    let mut output = vec![0.0f32; output_time * weights.output];
    let mut tap_weight = vec![0.0f32; weights.input * weights.output];
    let mut tap_output = vec![0.0f32; time * weights.output];
    for tap in 0..weights.kernel {
        for input_channel in 0..weights.input {
            for output_channel in 0..weights.output {
                tap_weight[input_channel * weights.output + output_channel] = weights.weight
                    [(input_channel * weights.output + output_channel) * weights.kernel + tap];
            }
        }
        compute.gemm_f32(
            time,
            weights.output,
            weights.input,
            &frame_major,
            &tap_weight,
            None,
            &mut tap_output,
        )?;
        for source_time in 0..time {
            let uncropped = source_time * weights.stride + tap;
            if uncropped < weights.padding {
                continue;
            }
            let destination_time = uncropped - weights.padding;
            if destination_time >= output_time {
                continue;
            }
            for output_channel in 0..weights.output {
                output[destination_time * weights.output + output_channel] +=
                    tap_output[source_time * weights.output + output_channel];
            }
        }
    }
    for row in output.chunks_exact_mut(weights.output) {
        for (value, bias) in row.iter_mut().zip(&weights.bias) {
            *value += *bias;
        }
    }
    Ok((
        frame_to_channel(&output, output_time, weights.output),
        output_time,
    ))
}

fn linear_forward(
    input: &[f32],
    rows: usize,
    weights: &Linear,
    compute: &Compute,
) -> Result<Vec<f32>> {
    if input.len() != rows * weights.input {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: linear input has {} values, expected {rows}x{}",
            input.len(),
            weights.input
        )));
    }
    let mut output = vec![0.0f32; rows * weights.output];
    compute.gemm_f32(
        rows,
        weights.output,
        weights.input,
        input,
        &weights.weight_t,
        Some(&weights.bias),
        &mut output,
    )?;
    Ok(output)
}

fn prosody_features(pcm: &[f32], weights: &EncoderWeights) -> Result<Vec<f32>> {
    if weights.hann_window.len() != MEL_WINDOW || weights.mel_basis.len() != MEL_BINS * 513 {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: mel frontend buffers have hann={} mel={}, expected {MEL_WINDOW} and {}",
            weights.hann_window.len(),
            weights.mel_basis.len(),
            MEL_BINS * 513
        )));
    }
    let padded = reflect_pad_1d(pcm, REFLECT_PAD)?;
    if padded.len() < MEL_N_FFT {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: padded PCM is shorter than n_fft {MEL_N_FFT}"
        )));
    }
    let frames = (padded.len() - MEL_N_FFT) / MEL_HOP + 1;
    let plan = RealFftPlan::new(MEL_N_FFT);
    let mut frame = vec![0.0f32; MEL_N_FFT];
    let mut output = vec![0.0f32; frames * PROSODY_BINS];
    let window_offset = (MEL_N_FFT - MEL_WINDOW) / 2;
    for index in 0..frames {
        frame.fill(0.0);
        let source = index * MEL_HOP;
        for sample in 0..MEL_WINDOW {
            frame[window_offset + sample] =
                padded[source + window_offset + sample] * weights.hann_window[sample];
        }
        let spectrum = plan.forward(&frame);
        let magnitude: Vec<f32> = spectrum
            .iter()
            .map(|value| (value.re * value.re + value.im * value.im + 1.0e-9).sqrt())
            .collect();
        for mel in 0..PROSODY_BINS {
            let basis = &weights.mel_basis[mel * 513..(mel + 1) * 513];
            let value = basis
                .iter()
                .zip(&magnitude)
                .map(|(left, right)| left * right)
                .sum::<f32>();
            output[index * PROSODY_BINS + mel] = value.max(1.0e-5).ln();
        }
    }
    reject_non_finite("prosody frontend", &output)?;
    Ok(output)
}

fn reflect_pad_1d(input: &[f32], padding: usize) -> Result<Vec<f32>> {
    if input.len() <= padding {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: reflect padding {padding} requires input length > {padding}, got {}",
            input.len()
        )));
    }
    let mut output = Vec::with_capacity(input.len() + 2 * padding);
    for index in (1..=padding).rev() {
        output.push(input[index]);
    }
    output.extend_from_slice(input);
    for index in 0..padding {
        output.push(input[input.len() - 2 - index]);
    }
    Ok(output)
}

fn replicate_pad_channel_major(
    input: &[f32],
    channels: usize,
    time: usize,
    left: usize,
    right: usize,
) -> Result<Vec<f32>> {
    if time == 0 || input.len() != channels * time {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: replicate padding input has {} values for {channels}x{time}",
            input.len()
        )));
    }
    let output_time = left + time + right;
    let mut output = vec![0.0f32; channels * output_time];
    for channel in 0..channels {
        let source = &input[channel * time..(channel + 1) * time];
        let target = &mut output[channel * output_time..(channel + 1) * output_time];
        target[..left].fill(source[0]);
        target[left..left + time].copy_from_slice(source);
        target[left + time..].fill(source[time - 1]);
    }
    Ok(output)
}

fn repeat_depthwise_kernel(kernel: &[f32], channels: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(kernel.len() * channels);
    for _ in 0..channels {
        output.extend_from_slice(kernel);
    }
    output
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
            "{LABEL}: {label} contains non-finite value {value} at index {index}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_padding_matches_torch_edge_exclusion() {
        assert_eq!(
            reflect_pad_1d(&[1.0, 2.0, 3.0, 4.0], 2).unwrap(),
            vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
    }

    #[test]
    fn layout_round_trip_is_exact() {
        let frame_major = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let channel_major = frame_to_channel(&frame_major, 3, 2);
        assert_eq!(channel_major, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        assert_eq!(channel_to_frame(&channel_major, 2, 3), frame_major);
    }
}
