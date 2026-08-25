//! Native MP-SENet forward graph.
//!
//! Learned reductions are lowered to the shared [`Compute`] seam: 2-D
//! convolutions use chunked im2col + GEMM, Transformer attention uses
//! GEMM/softmax (or the existing device-resident attention chain), GRU gates
//! use GEMM, and both LayerNorm and InstanceNorm reductions use LayerNorm.
//! Indexing, layout changes, element-wise activations, STFT/iSTFT and the small
//! per-channel affine after InstanceNorm remain host glue.  A non-CPU backend
//! is selected once by the caller and no learned reduction falls back to CPU.

use vokra_core::ir::graph::{IstftAttrs, StftAttrs};
use vokra_core::{Result, VokraError};
use vokra_ops::{Spectrogram, istft, stft};

use crate::compute::Compute;

use super::weights::{
    Attention, BiGru, Conv2d, ConvNormPrelu, DecoderStem, DenseBlock, DenseEncoder, GruDirection,
    Linear, MaskDecoder, MpSenetWeights, Norm, PhaseDecoder, Transformer, TsBlock,
};
use super::{
    ATTENTION_HEADS, COMPRESS_FACTOR, DENSE_CHANNELS, GRU_HIDDEN, HOP_LENGTH, MASK_BETA, N_BINS,
    N_FFT,
};

const INSTANCE_NORM_EPS: f32 = 1.0e-5;
const LAYER_NORM_EPS: f32 = 1.0e-5;
const MAGNITUDE_EPS: f32 = 1.0e-9;
const PHASE_IMAG_EPS: f32 = 1.0e-10;
const PHASE_REAL_EPS: f32 = 1.0e-5;
const LEAKY_RELU_SLOPE: f32 = 0.01;
const IM2COL_CHUNK: usize = 4_096;

#[derive(Debug, Clone)]
struct Tensor4 {
    /// Contiguous NCHW storage.
    data: Vec<f32>,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
}

impl Tensor4 {
    fn new(
        data: Vec<f32>,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<Self> {
        let expected = checked_product("tensor4", &[batch, channels, height, width])?;
        if data.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "mp_senet: Tensor4 has {} values, expected {batch} x {channels} x {height} x {width} = {expected}",
                data.len()
            )));
        }
        Ok(Self {
            data,
            batch,
            channels,
            height,
            width,
        })
    }

    fn index(&self, batch: usize, channel: usize, height: usize, width: usize) -> usize {
        (((batch * self.channels + channel) * self.height + height) * self.width) + width
    }
}

#[derive(Debug, Clone, Copy)]
struct ConvParams {
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
    dilation_h: usize,
    dilation_w: usize,
}

impl Default for ConvParams {
    fn default() -> Self {
        Self {
            stride_h: 1,
            stride_w: 1,
            pad_top: 0,
            pad_bottom: 0,
            pad_left: 0,
            pad_right: 0,
            dilation_h: 1,
            dilation_w: 1,
        }
    }
}

pub(super) fn enhance_segment(
    compute: &Compute,
    weights: &MpSenetWeights,
    pcm: &[f32],
) -> Result<Vec<f32>> {
    let spectrum = stft(pcm, &StftAttrs::new(N_FFT, HOP_LENGTH))?;
    if spectrum.frames == 0 || spectrum.bins != N_BINS {
        return Err(VokraError::InvalidArgument(format!(
            "mp_senet: STFT emitted {} frames x {} bins, expected positive frames x {N_BINS}",
            spectrum.frames, spectrum.bins
        )));
    }

    let spatial = checked_product("frontend", &[spectrum.frames, spectrum.bins])?;
    let mut magnitude = vec![0.0f32; spatial];
    let mut features = vec![0.0f32; 2 * spatial];
    for index in 0..spatial {
        let real = spectrum.re[index];
        let imag = spectrum.im[index];
        let compressed = (real * real + imag * imag + MAGNITUDE_EPS)
            .sqrt()
            .powf(COMPRESS_FACTOR);
        let phase = (imag + PHASE_IMAG_EPS).atan2(real + PHASE_REAL_EPS);
        magnitude[index] = compressed;
        features[index] = compressed;
        features[spatial + index] = phase;
    }

    let mut hidden = Tensor4::new(features, 1, 2, spectrum.frames, spectrum.bins)?;
    hidden = dense_encoder(compute, hidden, &weights.encoder)?;
    for block in &weights.transformers {
        hidden = ts_block(compute, hidden, block)?;
    }

    let mask = mask_decoder(compute, hidden.clone(), &weights.mask)?;
    let phase = phase_decoder(compute, hidden, &weights.phase)?;
    if mask.batch != 1
        || phase.batch != 1
        || mask.channels != 1
        || phase.channels != 1
        || mask.height != spectrum.frames
        || phase.height != spectrum.frames
        || mask.width != N_BINS
        || phase.width != N_BINS
    {
        return Err(VokraError::ModelLoad(format!(
            "mp_senet: decoder shape mismatch: mask={:?}, phase={:?}, frontend={}x{}",
            (mask.batch, mask.channels, mask.height, mask.width),
            (phase.batch, phase.channels, phase.height, phase.width),
            spectrum.frames,
            spectrum.bins
        )));
    }

    let mut prediction = Spectrogram {
        frames: spectrum.frames,
        bins: spectrum.bins,
        re: vec![0.0f32; spatial],
        im: vec![0.0f32; spatial],
    };
    for frame in 0..spectrum.frames {
        for bin in 0..spectrum.bins {
            let index = frame * spectrum.bins + bin;
            let gain = MASK_BETA
                * sigmoid(
                    weights.mask.sigmoid_slope[bin] * mask.data[mask.index(0, 0, frame, bin)],
                );
            let decompressed = (magnitude[index] * gain).powf(1.0 / COMPRESS_FACTOR);
            let angle = phase.data[phase.index(0, 0, frame, bin)];
            prediction.re[index] = decompressed * angle.cos();
            prediction.im[index] = decompressed * angle.sin();
        }
    }
    if prediction
        .re
        .iter()
        .chain(&prediction.im)
        .any(|value| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(
            "mp_senet: decoder emitted a non-finite spectrogram".to_owned(),
        ));
    }

    let output = istft(&prediction, &IstftAttrs::new(N_FFT, HOP_LENGTH))?;
    if output.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "mp_senet: iSTFT emitted a non-finite waveform".to_owned(),
        ));
    }
    Ok(output)
}

fn dense_encoder(compute: &Compute, input: Tensor4, weights: &DenseEncoder) -> Result<Tensor4> {
    let mut hidden = conv_norm_prelu(compute, input, &weights.input, ConvParams::default())?;
    hidden = dense_block(compute, hidden, &weights.dense)?;
    conv_norm_prelu(
        compute,
        hidden,
        &weights.downsample,
        ConvParams {
            stride_w: 2,
            pad_left: 1,
            pad_right: 1,
            ..ConvParams::default()
        },
    )
}

fn conv_norm_prelu(
    compute: &Compute,
    input: Tensor4,
    weights: &ConvNormPrelu,
    params: ConvParams,
) -> Result<Tensor4> {
    let mut output = conv2d(compute, &input, &weights.conv, params)?;
    instance_norm(compute, &mut output, &weights.norm)?;
    prelu(&mut output, &weights.slope)?;
    Ok(output)
}

fn dense_block(compute: &Compute, input: Tensor4, weights: &DenseBlock) -> Result<Tensor4> {
    let mut skip = input;
    let mut last = None;
    for layer in &weights.layers {
        let mut hidden = conv2d(
            compute,
            &skip,
            &layer.conv,
            ConvParams {
                pad_top: layer.dilation_h,
                pad_left: 1,
                pad_right: 1,
                dilation_h: layer.dilation_h,
                ..ConvParams::default()
            },
        )?;
        instance_norm(compute, &mut hidden, &layer.norm)?;
        prelu(&mut hidden, &layer.slope)?;
        skip = concat_channels(&hidden, &skip)?;
        last = Some(hidden);
    }
    last.ok_or_else(|| VokraError::ModelLoad("mp_senet: empty DenseBlock".to_owned()))
}

fn ts_block(compute: &Compute, input: Tensor4, weights: &TsBlock) -> Result<Tensor4> {
    if input.channels != DENSE_CHANNELS {
        return Err(VokraError::ModelLoad(format!(
            "mp_senet: TS block received {} channels, expected {DENSE_CHANNELS}",
            input.channels
        )));
    }
    let time_sequence = nchw_to_time_sequence(&input);
    let mut time_output = transformer(
        compute,
        &time_sequence,
        input.batch * input.width,
        input.height,
        &weights.time,
    )?;
    add_in_place(&mut time_output, &time_sequence, "time outer residual")?;

    let frequency_sequence = time_to_frequency_sequence(
        &time_output,
        input.batch,
        input.height,
        input.width,
        input.channels,
    )?;
    let mut frequency_output = transformer(
        compute,
        &frequency_sequence,
        input.batch * input.height,
        input.width,
        &weights.frequency,
    )?;
    add_in_place(
        &mut frequency_output,
        &frequency_sequence,
        "frequency outer residual",
    )?;
    frequency_sequence_to_nchw(
        frequency_output,
        input.batch,
        input.height,
        input.width,
        input.channels,
    )
}

/// Reproduce the released package's `batch_first=False` quirk exactly.
/// `sequence` is the first axis and `batch` the second, even though the
/// package constructed its source tensor as `[b * axis, other_axis, c]`.
fn transformer(
    compute: &Compute,
    input: &[f32],
    sequence: usize,
    batch: usize,
    weights: &Transformer,
) -> Result<Vec<f32>> {
    let rows = checked_product("transformer rows", &[sequence, batch])?;
    if input.len() != rows * DENSE_CHANNELS {
        return Err(VokraError::InvalidArgument(format!(
            "mp_senet: Transformer input has {} values, expected {sequence} x {batch} x {DENSE_CHANNELS}",
            input.len()
        )));
    }
    let normalized = layer_norm(compute, input, rows, &weights.norm1)?;
    let attention = multihead_attention(compute, &normalized, sequence, batch, &weights.attention)?;
    let mut hidden = input.to_vec();
    add_in_place(&mut hidden, &attention, "attention residual")?;

    let normalized = layer_norm(compute, &hidden, rows, &weights.norm2)?;
    let mut recurrent = bigru(compute, &normalized, sequence, batch, &weights.gru)?;
    for value in &mut recurrent {
        if *value < 0.0 {
            *value *= LEAKY_RELU_SLOPE;
        }
    }
    let projected = linear(compute, &recurrent, rows, &weights.linear)?;
    add_in_place(&mut hidden, &projected, "FFN residual")?;
    layer_norm(compute, &hidden, rows, &weights.norm3)
}

fn multihead_attention(
    compute: &Compute,
    input: &[f32],
    sequence: usize,
    batch: usize,
    weights: &Attention,
) -> Result<Vec<f32>> {
    let rows = checked_product("attention rows", &[sequence, batch])?;
    let mut qkv = vec![0.0f32; rows * 3 * DENSE_CHANNELS];
    compute.gemm_f32(
        rows,
        3 * DENSE_CHANNELS,
        DENSE_CHANNELS,
        input,
        &weights.in_weight_t,
        Some(&weights.in_bias),
        &mut qkv,
    )?;

    let head_dim = DENSE_CHANNELS / ATTENTION_HEADS;
    let scale = (head_dim as f32).powf(-0.5);
    let mut combined = vec![0.0f32; rows * DENSE_CHANNELS];
    for batch_index in 0..batch {
        let mut x = vec![0.0f32; sequence * DENSE_CHANNELS];
        let mut query = vec![0.0f32; sequence * DENSE_CHANNELS];
        let mut key = vec![0.0f32; sequence * DENSE_CHANNELS];
        let mut value = vec![0.0f32; sequence * DENSE_CHANNELS];
        for position in 0..sequence {
            let source_row = position * batch + batch_index;
            x[position * DENSE_CHANNELS..(position + 1) * DENSE_CHANNELS].copy_from_slice(
                &input[source_row * DENSE_CHANNELS..(source_row + 1) * DENSE_CHANNELS],
            );
            let source = source_row * 3 * DENSE_CHANNELS;
            query[position * DENSE_CHANNELS..(position + 1) * DENSE_CHANNELS]
                .copy_from_slice(&qkv[source..source + DENSE_CHANNELS]);
            key[position * DENSE_CHANNELS..(position + 1) * DENSE_CHANNELS]
                .copy_from_slice(&qkv[source + DENSE_CHANNELS..source + 2 * DENSE_CHANNELS]);
            value[position * DENSE_CHANNELS..(position + 1) * DENSE_CHANNELS]
                .copy_from_slice(&qkv[source + 2 * DENSE_CHANNELS..source + 3 * DENSE_CHANNELS]);
        }

        let batch_output = if compute.attention_is_fused() {
            let mut output = vec![0.0f32; sequence * DENSE_CHANNELS];
            compute.attn_f32(
                sequence,
                sequence,
                DENSE_CHANNELS,
                ATTENTION_HEADS,
                &x,
                &weights.q_weight_t,
                Some(&weights.in_bias[..DENSE_CHANNELS]),
                &key,
                &value,
                &weights.out_weight_t,
                Some(&weights.out_bias),
                scale,
                &mut output,
            )?;
            output
        } else {
            attention_heads(compute, &query, &key, &value, sequence, head_dim, scale)?
        };

        for position in 0..sequence {
            let destination_row = position * batch + batch_index;
            combined[destination_row * DENSE_CHANNELS..(destination_row + 1) * DENSE_CHANNELS]
                .copy_from_slice(
                    &batch_output[position * DENSE_CHANNELS..(position + 1) * DENSE_CHANNELS],
                );
        }
    }

    if compute.attention_is_fused() {
        return Ok(combined);
    }
    let mut output = vec![0.0f32; rows * DENSE_CHANNELS];
    compute.gemm_f32(
        rows,
        DENSE_CHANNELS,
        DENSE_CHANNELS,
        &combined,
        &weights.out_weight_t,
        Some(&weights.out_bias),
        &mut output,
    )?;
    Ok(output)
}

fn attention_heads(
    compute: &Compute,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sequence: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; sequence * DENSE_CHANNELS];
    for head in 0..ATTENTION_HEADS {
        let channel_start = head * head_dim;
        let mut q = vec![0.0f32; sequence * head_dim];
        let mut key_t = vec![0.0f32; head_dim * sequence];
        let mut v = vec![0.0f32; sequence * head_dim];
        for position in 0..sequence {
            for channel in 0..head_dim {
                q[position * head_dim + channel] =
                    query[position * DENSE_CHANNELS + channel_start + channel] * scale;
                key_t[channel * sequence + position] =
                    key[position * DENSE_CHANNELS + channel_start + channel];
                v[position * head_dim + channel] =
                    value[position * DENSE_CHANNELS + channel_start + channel];
            }
        }
        let mut scores = vec![0.0f32; sequence * sequence];
        compute.gemm_f32(sequence, sequence, head_dim, &q, &key_t, None, &mut scores)?;
        let mut probabilities = vec![0.0f32; scores.len()];
        compute.softmax_f32(&scores, &mut probabilities, sequence, sequence)?;
        let mut context = vec![0.0f32; sequence * head_dim];
        compute.gemm_f32(
            sequence,
            head_dim,
            sequence,
            &probabilities,
            &v,
            None,
            &mut context,
        )?;
        for position in 0..sequence {
            output[position * DENSE_CHANNELS + channel_start
                ..position * DENSE_CHANNELS + channel_start + head_dim]
                .copy_from_slice(&context[position * head_dim..(position + 1) * head_dim]);
        }
    }
    Ok(output)
}

fn bigru(
    compute: &Compute,
    input: &[f32],
    sequence: usize,
    batch: usize,
    weights: &BiGru,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; sequence * batch * 2 * GRU_HIDDEN];
    gru_direction(
        compute,
        input,
        sequence,
        batch,
        &weights.forward,
        false,
        &mut output,
    )?;
    gru_direction(
        compute,
        input,
        sequence,
        batch,
        &weights.reverse,
        true,
        &mut output,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn gru_direction(
    compute: &Compute,
    input: &[f32],
    sequence: usize,
    batch: usize,
    weights: &GruDirection,
    reverse: bool,
    output: &mut [f32],
) -> Result<()> {
    let rows = checked_product("GRU rows", &[sequence, batch])?;
    let gates = 3 * GRU_HIDDEN;
    let mut input_projection = vec![0.0f32; rows * gates];
    compute.gemm_f32(
        rows,
        gates,
        DENSE_CHANNELS,
        input,
        &weights.weight_ih_t,
        Some(&weights.bias_ih),
        &mut input_projection,
    )?;

    let mut hidden = vec![0.0f32; batch * GRU_HIDDEN];
    let mut recurrent = vec![0.0f32; batch * gates];
    let positions: Box<dyn Iterator<Item = usize>> = if reverse {
        Box::new((0..sequence).rev())
    } else {
        Box::new(0..sequence)
    };
    for position in positions {
        compute.gemm_f32(
            batch,
            gates,
            GRU_HIDDEN,
            &hidden,
            &weights.weight_hh_t,
            Some(&weights.bias_hh),
            &mut recurrent,
        )?;
        for batch_index in 0..batch {
            let input_base = (position * batch + batch_index) * gates;
            let recurrent_base = batch_index * gates;
            let hidden_base = batch_index * GRU_HIDDEN;
            let output_base = (position * batch + batch_index) * 2 * GRU_HIDDEN
                + usize::from(reverse) * GRU_HIDDEN;
            for channel in 0..GRU_HIDDEN {
                // PyTorch GRU gate order is reset, update, new.  The new gate
                // applies reset only to the recurrent affine (including b_hn).
                let reset = sigmoid(
                    input_projection[input_base + channel] + recurrent[recurrent_base + channel],
                );
                let update = sigmoid(
                    input_projection[input_base + GRU_HIDDEN + channel]
                        + recurrent[recurrent_base + GRU_HIDDEN + channel],
                );
                let candidate = (input_projection[input_base + 2 * GRU_HIDDEN + channel]
                    + reset * recurrent[recurrent_base + 2 * GRU_HIDDEN + channel])
                    .tanh();
                let previous = hidden[hidden_base + channel];
                let next = (1.0 - update) * candidate + update * previous;
                hidden[hidden_base + channel] = next;
                output[output_base + channel] = next;
            }
        }
    }
    Ok(())
}

fn linear(compute: &Compute, input: &[f32], rows: usize, weights: &Linear) -> Result<Vec<f32>> {
    if input.len() != rows * weights.input {
        return Err(VokraError::InvalidArgument(format!(
            "mp_senet: Linear input has {} values, expected {rows} x {}",
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

fn layer_norm(compute: &Compute, input: &[f32], rows: usize, weights: &Norm) -> Result<Vec<f32>> {
    if weights.gamma.len() != DENSE_CHANNELS || weights.beta.len() != DENSE_CHANNELS {
        return Err(VokraError::ModelLoad(
            "mp_senet: Transformer LayerNorm affine shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0f32; input.len()];
    compute.layer_norm_f32(
        input,
        &mut output,
        rows,
        DENSE_CHANNELS,
        &weights.gamma,
        &weights.beta,
        LAYER_NORM_EPS,
    )?;
    Ok(output)
}

fn mask_decoder(compute: &Compute, input: Tensor4, weights: &MaskDecoder) -> Result<Tensor4> {
    let hidden = dense_block(compute, input, &weights.dense)?;
    let hidden = decoder_stem(compute, hidden, &weights.stem)?;
    conv2d(compute, &hidden, &weights.output, ConvParams::default())
}

fn phase_decoder(compute: &Compute, input: Tensor4, weights: &PhaseDecoder) -> Result<Tensor4> {
    let hidden = dense_block(compute, input, &weights.dense)?;
    let hidden = decoder_stem(compute, hidden, &weights.stem)?;
    let real = conv2d(compute, &hidden, &weights.real, ConvParams::default())?;
    let imag = conv2d(compute, &hidden, &weights.imag, ConvParams::default())?;
    if real.batch != imag.batch
        || real.channels != imag.channels
        || real.height != imag.height
        || real.width != imag.width
    {
        return Err(VokraError::ModelLoad(
            "mp_senet: phase real/imag decoder shapes disagree".to_owned(),
        ));
    }
    let data = imag
        .data
        .iter()
        .zip(&real.data)
        .map(|(&imaginary, &real)| imaginary.atan2(real))
        .collect();
    Tensor4::new(data, real.batch, real.channels, real.height, real.width)
}

fn decoder_stem(compute: &Compute, input: Tensor4, weights: &DecoderStem) -> Result<Tensor4> {
    let expanded = conv2d(
        compute,
        &input,
        &weights.expand,
        ConvParams {
            pad_left: 1,
            pad_right: 1,
            ..ConvParams::default()
        },
    )?;
    let mut shuffled = pixel_shuffle_frequency(expanded, 2)?;
    instance_norm(compute, &mut shuffled, &weights.norm)?;
    prelu(&mut shuffled, &weights.slope)?;
    Ok(shuffled)
}

fn pixel_shuffle_frequency(input: Tensor4, ratio: usize) -> Result<Tensor4> {
    if ratio == 0 || input.channels % ratio != 0 {
        return Err(VokraError::ModelLoad(format!(
            "mp_senet: pixel shuffle ratio {ratio} does not divide {} channels",
            input.channels
        )));
    }
    let channels = input.channels / ratio;
    let width = input.width.checked_mul(ratio).ok_or_else(|| {
        VokraError::InvalidArgument("mp_senet: pixel shuffle width overflow".to_owned())
    })?;
    let mut output = vec![0.0f32; input.batch * channels * input.height * width];
    for batch in 0..input.batch {
        for channel in 0..channels {
            for height in 0..input.height {
                for source_width in 0..input.width {
                    for subpixel in 0..ratio {
                        let source_channel = subpixel * channels + channel;
                        let source = input.index(batch, source_channel, height, source_width);
                        let destination = (((batch * channels + channel) * input.height + height)
                            * width)
                            + source_width * ratio
                            + subpixel;
                        output[destination] = input.data[source];
                    }
                }
            }
        }
    }
    Tensor4::new(output, input.batch, channels, input.height, width)
}

fn conv2d(
    compute: &Compute,
    input: &Tensor4,
    weights: &Conv2d,
    params: ConvParams,
) -> Result<Tensor4> {
    if input.channels != weights.input
        || weights.weight.len()
            != weights.output * weights.input * weights.kernel_h * weights.kernel_w
        || weights.bias.len() != weights.output
        || params.stride_h == 0
        || params.stride_w == 0
        || params.dilation_h == 0
        || params.dilation_w == 0
    {
        return Err(VokraError::ModelLoad(format!(
            "mp_senet: Conv2d shape/attribute mismatch (input channels {}, weight {}->{}, kernel {}x{})",
            input.channels, weights.input, weights.output, weights.kernel_h, weights.kernel_w
        )));
    }
    let effective_h = (weights.kernel_h - 1)
        .checked_mul(params.dilation_h)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            VokraError::InvalidArgument("mp_senet: Conv2d height overflow".to_owned())
        })?;
    let effective_w = (weights.kernel_w - 1)
        .checked_mul(params.dilation_w)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| VokraError::InvalidArgument("mp_senet: Conv2d width overflow".to_owned()))?;
    let padded_h = input
        .height
        .checked_add(params.pad_top)
        .and_then(|value| value.checked_add(params.pad_bottom))
        .ok_or_else(|| {
            VokraError::InvalidArgument("mp_senet: Conv2d height overflow".to_owned())
        })?;
    let padded_w = input
        .width
        .checked_add(params.pad_left)
        .and_then(|value| value.checked_add(params.pad_right))
        .ok_or_else(|| VokraError::InvalidArgument("mp_senet: Conv2d width overflow".to_owned()))?;
    if padded_h < effective_h || padded_w < effective_w {
        return Err(VokraError::InvalidArgument(format!(
            "mp_senet: Conv2d effective kernel {effective_h}x{effective_w} exceeds padded input {padded_h}x{padded_w}"
        )));
    }
    let output_h = (padded_h - effective_h) / params.stride_h + 1;
    let output_w = (padded_w - effective_w) / params.stride_w + 1;
    let positions = checked_product("Conv2d positions", &[output_h, output_w])?;
    let patch = checked_product(
        "Conv2d patch",
        &[weights.input, weights.kernel_h, weights.kernel_w],
    )?;
    let output_len = checked_product(
        "Conv2d output",
        &[input.batch, weights.output, output_h, output_w],
    )?;
    let mut output = vec![0.0f32; output_len];

    for batch in 0..input.batch {
        for chunk_start in (0..positions).step_by(IM2COL_CHUNK) {
            let chunk = (positions - chunk_start).min(IM2COL_CHUNK);
            let mut columns = vec![0.0f32; patch * chunk];
            for input_channel in 0..weights.input {
                for kernel_h in 0..weights.kernel_h {
                    for kernel_w in 0..weights.kernel_w {
                        let patch_index = (input_channel * weights.kernel_h + kernel_h)
                            * weights.kernel_w
                            + kernel_w;
                        for local in 0..chunk {
                            let position = chunk_start + local;
                            let output_y = position / output_w;
                            let output_x = position % output_w;
                            let source_y =
                                output_y * params.stride_h + kernel_h * params.dilation_h;
                            let source_x =
                                output_x * params.stride_w + kernel_w * params.dilation_w;
                            if source_y >= params.pad_top
                                && source_x >= params.pad_left
                                && source_y - params.pad_top < input.height
                                && source_x - params.pad_left < input.width
                            {
                                let input_y = source_y - params.pad_top;
                                let input_x = source_x - params.pad_left;
                                columns[patch_index * chunk + local] =
                                    input.data[input.index(batch, input_channel, input_y, input_x)];
                            }
                        }
                    }
                }
            }
            let mut block = vec![0.0f32; weights.output * chunk];
            compute.gemm_f32(
                weights.output,
                chunk,
                patch,
                &weights.weight,
                &columns,
                None,
                &mut block,
            )?;
            for output_channel in 0..weights.output {
                let destination =
                    (batch * weights.output + output_channel) * positions + chunk_start;
                for local in 0..chunk {
                    output[destination + local] =
                        block[output_channel * chunk + local] + weights.bias[output_channel];
                }
            }
        }
    }
    Tensor4::new(output, input.batch, weights.output, output_h, output_w)
}

fn instance_norm(compute: &Compute, tensor: &mut Tensor4, weights: &Norm) -> Result<()> {
    if weights.gamma.len() != tensor.channels || weights.beta.len() != tensor.channels {
        return Err(VokraError::ModelLoad(format!(
            "mp_senet: InstanceNorm has {} gamma / {} beta values for {} channels",
            weights.gamma.len(),
            weights.beta.len(),
            tensor.channels
        )));
    }
    let positions = checked_product("InstanceNorm positions", &[tensor.height, tensor.width])?;
    let rows = checked_product("InstanceNorm rows", &[tensor.batch, tensor.channels])?;
    let unit_gamma = vec![1.0f32; positions];
    let zero_beta = vec![0.0f32; positions];
    let mut normalized = vec![0.0f32; tensor.data.len()];
    compute.layer_norm_f32(
        &tensor.data,
        &mut normalized,
        rows,
        positions,
        &unit_gamma,
        &zero_beta,
        INSTANCE_NORM_EPS,
    )?;
    for batch in 0..tensor.batch {
        for channel in 0..tensor.channels {
            let start = (batch * tensor.channels + channel) * positions;
            for value in &mut normalized[start..start + positions] {
                *value = *value * weights.gamma[channel] + weights.beta[channel];
            }
        }
    }
    tensor.data = normalized;
    Ok(())
}

fn prelu(tensor: &mut Tensor4, slope: &[f32]) -> Result<()> {
    if slope.len() != tensor.channels {
        return Err(VokraError::ModelLoad(format!(
            "mp_senet: PReLU has {} slopes for {} channels",
            slope.len(),
            tensor.channels
        )));
    }
    let positions = tensor.height * tensor.width;
    for batch in 0..tensor.batch {
        for channel in 0..tensor.channels {
            let start = (batch * tensor.channels + channel) * positions;
            for value in &mut tensor.data[start..start + positions] {
                if *value < 0.0 {
                    *value *= slope[channel];
                }
            }
        }
    }
    Ok(())
}

fn concat_channels(left: &Tensor4, right: &Tensor4) -> Result<Tensor4> {
    if left.batch != right.batch || left.height != right.height || left.width != right.width {
        return Err(VokraError::ModelLoad(
            "mp_senet: DenseBlock channel concatenation shape mismatch".to_owned(),
        ));
    }
    let channels = left.channels + right.channels;
    let spatial = left.height * left.width;
    let mut data = Vec::with_capacity(left.batch * channels * spatial);
    for batch in 0..left.batch {
        let left_start = batch * left.channels * spatial;
        data.extend_from_slice(&left.data[left_start..left_start + left.channels * spatial]);
        let right_start = batch * right.channels * spatial;
        data.extend_from_slice(&right.data[right_start..right_start + right.channels * spatial]);
    }
    Tensor4::new(data, left.batch, channels, left.height, left.width)
}

fn nchw_to_time_sequence(input: &Tensor4) -> Vec<f32> {
    let mut output = vec![0.0f32; input.data.len()];
    for batch in 0..input.batch {
        for frequency in 0..input.width {
            let sequence = batch * input.width + frequency;
            for time in 0..input.height {
                for channel in 0..input.channels {
                    output[(sequence * input.height + time) * input.channels + channel] =
                        input.data[input.index(batch, channel, time, frequency)];
                }
            }
        }
    }
    output
}

fn time_to_frequency_sequence(
    input: &[f32],
    batch: usize,
    time: usize,
    frequency: usize,
    channels: usize,
) -> Result<Vec<f32>> {
    if input.len() != batch * time * frequency * channels {
        return Err(VokraError::InvalidArgument(
            "mp_senet: time sequence reshape length mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0f32; input.len()];
    for batch_index in 0..batch {
        for time_index in 0..time {
            for frequency_index in 0..frequency {
                for channel in 0..channels {
                    let source = (((batch_index * frequency + frequency_index) * time
                        + time_index)
                        * channels)
                        + channel;
                    let destination = (((batch_index * time + time_index) * frequency
                        + frequency_index)
                        * channels)
                        + channel;
                    output[destination] = input[source];
                }
            }
        }
    }
    Ok(output)
}

fn frequency_sequence_to_nchw(
    input: Vec<f32>,
    batch: usize,
    time: usize,
    frequency: usize,
    channels: usize,
) -> Result<Tensor4> {
    if input.len() != batch * time * frequency * channels {
        return Err(VokraError::InvalidArgument(
            "mp_senet: frequency sequence reshape length mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0f32; input.len()];
    for batch_index in 0..batch {
        for channel in 0..channels {
            for time_index in 0..time {
                for frequency_index in 0..frequency {
                    let source = (((batch_index * time + time_index) * frequency
                        + frequency_index)
                        * channels)
                        + channel;
                    let destination = (((batch_index * channels + channel) * time + time_index)
                        * frequency)
                        + frequency_index;
                    output[destination] = input[source];
                }
            }
        }
    }
    Tensor4::new(output, batch, channels, time, frequency)
}

fn add_in_place(left: &mut [f32], right: &[f32], label: &str) -> Result<()> {
    if left.len() != right.len() {
        return Err(VokraError::ModelLoad(format!(
            "mp_senet: {label} length mismatch: {} != {}",
            left.len(),
            right.len()
        )));
    }
    for (left, &right) in left.iter_mut().zip(right) {
        *left += right;
    }
    Ok(())
}

fn checked_product(label: &str, dimensions: &[usize]) -> Result<usize> {
    dimensions.iter().try_fold(1usize, |product, &dimension| {
        product.checked_mul(dimension).ok_or_else(|| {
            VokraError::InvalidArgument(format!("mp_senet: {label} element count overflow"))
        })
    })
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_axis_permutations_round_trip() {
        let tensor = Tensor4::new((0..24).map(|value| value as f32).collect(), 1, 2, 3, 4).unwrap();
        let time = nchw_to_time_sequence(&tensor);
        let frequency = time_to_frequency_sequence(&time, 1, 3, 4, 2).unwrap();
        let restored = frequency_sequence_to_nchw(frequency, 1, 3, 4, 2).unwrap();
        assert_eq!(restored.data, tensor.data);
    }

    #[test]
    fn subpixel_frequency_order_matches_pytorch_view_permute() {
        let input = Tensor4::new(vec![10.0, 11.0, 20.0, 21.0], 1, 2, 1, 2).unwrap();
        let output = pixel_shuffle_frequency(input, 2).unwrap();
        assert_eq!((output.channels, output.height, output.width), (1, 1, 4));
        assert_eq!(output.data, vec![10.0, 20.0, 11.0, 21.0]);
    }
}
