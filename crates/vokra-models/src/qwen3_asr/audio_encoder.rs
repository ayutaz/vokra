//! Native bounded-memory Qwen3-ASR audio tower.
//!
//! Every learned operation is dispatched through one selected [`Compute`]
//! backend. Conv2D is lowered to im2col + GEMM; padding, layout transforms,
//! sinusoidal positions, residual additions and attention segmentation are
//! scalar host glue rather than hidden learned CPU fallbacks.

use std::sync::Mutex;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufTensorInfo;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::mapped_weights::{lock_scratch, transpose_widen, widen_into};

use super::frontend::{self, CONV_CHUNK_FRAMES, N_MELS, Qwen3AsrFeatures};
use super::weights::{Qwen3AsrConvDescriptors, Qwen3AsrMappedDescriptors};
use super::{Qwen3AsrCheckpoint, Qwen3AsrConfig};

/// Complete learned-op set of the Qwen3-ASR audio tower.
pub const QWEN3_ASR_AUDIO_HOT_OPS: &[HotOp] =
    &[HotOp::Gemm, HotOp::Softmax, HotOp::LayerNorm, HotOp::Gelu];

/// Projected audio-prefix embeddings consumed by the Qwen3 text decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3AsrAudioEmbeddings {
    values: Vec<f32>,
    frames: usize,
    hidden_size: usize,
}

impl Qwen3AsrAudioEmbeddings {
    /// Row-major `[frames, hidden_size]` projected audio values.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Number of audio-prefix rows.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    /// Width expected by the selected Qwen3 text decoder.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[derive(Default)]
pub(super) struct Qwen3AsrAudioRuntime {
    convolution: Mutex<ConvolutionBank>,
    layer: Mutex<AudioLayerBlock>,
    post: Mutex<AudioPostBlock>,
}

#[derive(Default)]
struct DenseLinear {
    weight_t: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Default)]
struct ConvolutionBank {
    ready: bool,
    conv1: DenseLinear,
    conv2: DenseLinear,
    conv3: DenseLinear,
    conv_out_weight_t: Vec<f32>,
}

#[derive(Default)]
struct AudioLayerBlock {
    self_attn_norm_weight: Vec<f32>,
    self_attn_norm_bias: Vec<f32>,
    q: DenseLinear,
    k: DenseLinear,
    v: DenseLinear,
    out: DenseLinear,
    final_norm_weight: Vec<f32>,
    final_norm_bias: Vec<f32>,
    fc1: DenseLinear,
    fc2: DenseLinear,
}

#[derive(Default)]
struct AudioPostBlock {
    ready: bool,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    proj1: DenseLinear,
    proj2: DenseLinear,
}

#[derive(Default)]
struct AudioStepScratch {
    norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    query: Vec<f32>,
    key_t: Vec<f32>,
    value: Vec<f32>,
    scores: Vec<f32>,
    probabilities: Vec<f32>,
    attended: Vec<f32>,
    attention: Vec<f32>,
    projected: Vec<f32>,
    ffn: Vec<f32>,
    activated: Vec<f32>,
    residual: Vec<f32>,
}

pub(super) fn encode(
    checkpoint: &Qwen3AsrCheckpoint,
    backend: BackendKind,
    runtime: &Qwen3AsrAudioRuntime,
    pcm: &[f32],
) -> Result<Qwen3AsrAudioEmbeddings> {
    let mapped = checkpoint.mapped()?;
    let config = mapped.config();
    let compute = Compute::for_backend(backend, QWEN3_ASR_AUDIO_HOT_OPS)?;
    let features = frontend::extract(pcm)?;

    let mut convolution = lock_scratch(&runtime.convolution, mapped.mapped_model())?;
    if !convolution.ready {
        materialize_convolution(mapped, config, &mut convolution)?;
        convolution.ready = true;
    }
    let (mut hidden, segments) = convolutional_stem(
        &compute,
        &features,
        config,
        &convolution,
        mapped.mapped_model().name,
    )?;
    drop(convolution);

    let mut scratch = AudioStepScratch::default();
    let mut layer_block = lock_scratch(&runtime.layer, mapped.mapped_model())?;
    for layer in 0..config.audio.n_layer as usize {
        materialize_audio_layer(mapped, layer, config, &mut layer_block)?;
        encoder_layer(
            &compute,
            &mut hidden,
            &segments,
            config,
            &layer_block,
            &mut scratch,
            mapped.mapped_model().name,
        )?;
    }
    drop(layer_block);

    let mut post = lock_scratch(&runtime.post, mapped.mapped_model())?;
    if !post.ready {
        materialize_audio_post(mapped, config, &mut post)?;
        post.ready = true;
    }
    let values = project_audio(
        &compute,
        &hidden,
        config,
        &post,
        &mut scratch,
        mapped.mapped_model().name,
    )?;
    let frames = values.len() / config.audio.output_dim as usize;
    Ok(Qwen3AsrAudioEmbeddings {
        values,
        frames,
        hidden_size: config.audio.output_dim as usize,
    })
}

fn materialize_convolution(
    mapped: &Qwen3AsrMappedDescriptors,
    config: Qwen3AsrConfig,
    bank: &mut ConvolutionBank,
) -> Result<()> {
    let channels = config.audio.downsample_hidden_size as usize;
    materialize_conv(mapped, mapped.convolution(1), channels, 1, &mut bank.conv1)?;
    materialize_conv(
        mapped,
        mapped.convolution(2),
        channels,
        channels,
        &mut bank.conv2,
    )?;
    materialize_conv(
        mapped,
        mapped.convolution(3),
        channels,
        channels,
        &mut bank.conv3,
    )?;
    transpose_tensor(
        mapped,
        mapped.conv_out(),
        config.audio.d_model as usize,
        channels * 16,
        &mut bank.conv_out_weight_t,
    )
}

fn materialize_conv(
    mapped: &Qwen3AsrMappedDescriptors,
    descriptors: Qwen3AsrConvDescriptors<'_>,
    out_channels: usize,
    in_channels: usize,
    output: &mut DenseLinear,
) -> Result<()> {
    transpose_tensor(
        mapped,
        descriptors.weight,
        out_channels,
        in_channels * 3 * 3,
        &mut output.weight_t,
    )?;
    widen_tensor(mapped, descriptors.bias, &mut output.bias)
}

fn convolutional_stem(
    compute: &Compute,
    features: &Qwen3AsrFeatures,
    config: Qwen3AsrConfig,
    bank: &ConvolutionBank,
    label: &str,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let frames = features.frames;
    let chunks = frames.div_ceil(CONV_CHUNK_FRAMES);
    let padded_width = frames.min(CONV_CHUNK_FRAMES);
    let padded_after_cnn = padded_width.div_ceil(8);
    let d_model = config.audio.d_model as usize;
    let channels = config.audio.downsample_hidden_size as usize;
    let expected_rows = frontend::encoded_frames(frames);
    let mut hidden = Vec::with_capacity(expected_rows * d_model);

    for chunk in 0..chunks {
        let frame_start = chunk * CONV_CHUNK_FRAMES;
        let actual_width = (frames - frame_start).min(CONV_CHUNK_FRAMES);
        let actual_after_cnn = actual_width.div_ceil(8);
        let mut input = vec![0.0; N_MELS * padded_width];
        for mel in 0..N_MELS {
            let source = mel * frames + frame_start;
            let target = mel * padded_width;
            input[target..target + actual_width]
                .copy_from_slice(&features.values[source..source + actual_width]);
        }

        let (stage1, height1, width1) = conv2d_stride2_gelu(
            compute,
            &input,
            1,
            N_MELS,
            padded_width,
            channels,
            &bank.conv1,
            label,
        )?;
        let (stage2, height2, width2) = conv2d_stride2_gelu(
            compute,
            &stage1,
            channels,
            height1,
            width1,
            channels,
            &bank.conv2,
            label,
        )?;
        let (stage3, height3, width3) = conv2d_stride2_gelu(
            compute,
            &stage2,
            channels,
            height2,
            width2,
            channels,
            &bank.conv3,
            label,
        )?;
        if height3 != 16 || width3 != padded_after_cnn {
            return Err(VokraError::ModelLoad(format!(
                "{label}: convolutional stem produced [C,{height3},{width3}], expected [C,16,{padded_after_cnn}]"
            )));
        }

        let flattened_width = channels * height3;
        let mut flattened = vec![0.0; width3 * flattened_width];
        for time in 0..width3 {
            for channel in 0..channels {
                for frequency in 0..height3 {
                    flattened[time * flattened_width + channel * height3 + frequency] =
                        stage3[(channel * height3 + frequency) * width3 + time];
                }
            }
        }
        let mut embedded = vec![0.0; width3 * d_model];
        compute.gemm_f32(
            width3,
            d_model,
            flattened_width,
            &flattened,
            &bank.conv_out_weight_t,
            None,
            &mut embedded,
        )?;
        add_sinusoidal_positions(&mut embedded, width3, d_model, label)?;
        hidden.extend_from_slice(&embedded[..actual_after_cnn * d_model]);
    }

    if hidden.len() != expected_rows * d_model {
        return Err(VokraError::ModelLoad(format!(
            "{label}: convolutional stem retained {} rows, expected {expected_rows}",
            hidden.len() / d_model
        )));
    }
    let base_window = config.audio.n_window as usize * 2;
    let infer_window = config.audio.n_window_infer as usize;
    if base_window == 0 || !infer_window.is_multiple_of(base_window) {
        return Err(VokraError::ModelLoad(format!(
            "{label}: invalid attention window ratio {infer_window}/{base_window}"
        )));
    }
    let segment_width = padded_after_cnn * (infer_window / base_window);
    let mut segments = Vec::new();
    let mut remaining = expected_rows;
    while remaining != 0 {
        let length = remaining.min(segment_width);
        segments.push(length);
        remaining -= length;
    }
    reject_non_finite(label, "convolutional embeddings", &hidden)?;
    Ok((hidden, segments))
}

#[allow(clippy::too_many_arguments)]
fn conv2d_stride2_gelu(
    compute: &Compute,
    input: &[f32],
    in_channels: usize,
    input_height: usize,
    input_width: usize,
    out_channels: usize,
    linear: &DenseLinear,
    label: &str,
) -> Result<(Vec<f32>, usize, usize)> {
    let expected = in_channels
        .checked_mul(input_height)
        .and_then(|value| value.checked_mul(input_width))
        .ok_or_else(|| VokraError::InvalidArgument(format!("{label}: conv input size overflow")))?;
    if input.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: conv input has {} values, expected {expected}",
            input.len()
        )));
    }
    let output_height = input_height.div_ceil(2);
    let output_width = input_width.div_ceil(2);
    let rows = output_height * output_width;
    let columns = in_channels * 3 * 3;
    let mut im2col = vec![0.0; rows * columns];
    for out_y in 0..output_height {
        for out_x in 0..output_width {
            let row = out_y * output_width + out_x;
            for channel in 0..in_channels {
                for kernel_y in 0..3 {
                    let input_y = out_y * 2 + kernel_y;
                    if input_y == 0 || input_y > input_height {
                        continue;
                    }
                    let input_y = input_y - 1;
                    for kernel_x in 0..3 {
                        let input_x = out_x * 2 + kernel_x;
                        if input_x == 0 || input_x > input_width {
                            continue;
                        }
                        let input_x = input_x - 1;
                        let column = (channel * 3 + kernel_y) * 3 + kernel_x;
                        im2col[row * columns + column] =
                            input[(channel * input_height + input_y) * input_width + input_x];
                    }
                }
            }
        }
    }
    let mut spatial = vec![0.0; rows * out_channels];
    compute.gemm_f32(
        rows,
        out_channels,
        columns,
        &im2col,
        &linear.weight_t,
        Some(&linear.bias),
        &mut spatial,
    )?;
    let mut activated = vec![0.0; spatial.len()];
    compute.gelu_f32(&spatial, &mut activated)?;
    let mut output = vec![0.0; activated.len()];
    for out_y in 0..output_height {
        for out_x in 0..output_width {
            let row = out_y * output_width + out_x;
            for channel in 0..out_channels {
                output[(channel * output_height + out_y) * output_width + out_x] =
                    activated[row * out_channels + channel];
            }
        }
    }
    Ok((output, output_height, output_width))
}

fn add_sinusoidal_positions(
    values: &mut [f32],
    rows: usize,
    columns: usize,
    label: &str,
) -> Result<()> {
    if columns < 4 || !columns.is_multiple_of(2) || values.len() != rows * columns {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: sinusoidal position shape mismatch: values={}, rows={rows}, columns={columns}",
            values.len()
        )));
    }
    let half = columns / 2;
    let increment = 10_000.0_f32.ln() / (half - 1) as f32;
    for row in 0..rows {
        for dimension in 0..half {
            let inverse_timescale = (-(dimension as f32) * increment).exp();
            let angle = row as f32 * inverse_timescale;
            values[row * columns + dimension] += angle.sin();
            values[row * columns + half + dimension] += angle.cos();
        }
    }
    Ok(())
}

fn materialize_audio_layer(
    mapped: &Qwen3AsrMappedDescriptors,
    layer: usize,
    config: Qwen3AsrConfig,
    block: &mut AudioLayerBlock,
) -> Result<()> {
    let descriptors = mapped.audio_layer(layer);
    let hidden = config.audio.d_model as usize;
    let ffn = config.audio.ffn_dim as usize;
    widen_tensor(
        mapped,
        descriptors.self_attn_norm_weight,
        &mut block.self_attn_norm_weight,
    )?;
    widen_tensor(
        mapped,
        descriptors.self_attn_norm_bias,
        &mut block.self_attn_norm_bias,
    )?;
    materialize_linear(
        mapped,
        descriptors.q_weight,
        descriptors.q_bias,
        hidden,
        hidden,
        &mut block.q,
    )?;
    materialize_linear(
        mapped,
        descriptors.k_weight,
        descriptors.k_bias,
        hidden,
        hidden,
        &mut block.k,
    )?;
    materialize_linear(
        mapped,
        descriptors.v_weight,
        descriptors.v_bias,
        hidden,
        hidden,
        &mut block.v,
    )?;
    materialize_linear(
        mapped,
        descriptors.out_weight,
        descriptors.out_bias,
        hidden,
        hidden,
        &mut block.out,
    )?;
    widen_tensor(
        mapped,
        descriptors.final_norm_weight,
        &mut block.final_norm_weight,
    )?;
    widen_tensor(
        mapped,
        descriptors.final_norm_bias,
        &mut block.final_norm_bias,
    )?;
    materialize_linear(
        mapped,
        descriptors.fc1_weight,
        descriptors.fc1_bias,
        ffn,
        hidden,
        &mut block.fc1,
    )?;
    materialize_linear(
        mapped,
        descriptors.fc2_weight,
        descriptors.fc2_bias,
        hidden,
        ffn,
        &mut block.fc2,
    )
}

fn materialize_linear(
    mapped: &Qwen3AsrMappedDescriptors,
    weight: &GgufTensorInfo,
    bias: &GgufTensorInfo,
    output: usize,
    input: usize,
    linear: &mut DenseLinear,
) -> Result<()> {
    transpose_tensor(mapped, weight, output, input, &mut linear.weight_t)?;
    widen_tensor(mapped, bias, &mut linear.bias)
}

#[allow(clippy::too_many_arguments)]
fn encoder_layer(
    compute: &Compute,
    hidden: &mut [f32],
    segments: &[usize],
    config: Qwen3AsrConfig,
    block: &AudioLayerBlock,
    scratch: &mut AudioStepScratch,
    label: &str,
) -> Result<()> {
    let columns = config.audio.d_model as usize;
    let rows = hidden.len() / columns;
    let ffn = config.audio.ffn_dim as usize;
    resize_zero(&mut scratch.norm, hidden.len());
    resize_zero(&mut scratch.q, hidden.len());
    resize_zero(&mut scratch.k, hidden.len());
    resize_zero(&mut scratch.v, hidden.len());
    resize_zero(&mut scratch.attention, hidden.len());
    resize_zero(&mut scratch.projected, hidden.len());
    resize_zero(&mut scratch.ffn, rows * ffn);
    resize_zero(&mut scratch.activated, rows * ffn);
    resize_zero(&mut scratch.residual, hidden.len());

    compute.layer_norm_f32(
        hidden,
        &mut scratch.norm,
        rows,
        columns,
        &block.self_attn_norm_weight,
        &block.self_attn_norm_bias,
        config.audio.layer_norm_eps,
    )?;
    compute.gemm_f32(
        rows,
        columns,
        columns,
        &scratch.norm,
        &block.q.weight_t,
        Some(&block.q.bias),
        &mut scratch.q,
    )?;
    compute.gemm_f32(
        rows,
        columns,
        columns,
        &scratch.norm,
        &block.k.weight_t,
        Some(&block.k.bias),
        &mut scratch.k,
    )?;
    compute.gemm_f32(
        rows,
        columns,
        columns,
        &scratch.norm,
        &block.v.weight_t,
        Some(&block.v.bias),
        &mut scratch.v,
    )?;
    segmented_attention(compute, segments, config, scratch, label)?;
    compute.gemm_f32(
        rows,
        columns,
        columns,
        &scratch.attention,
        &block.out.weight_t,
        Some(&block.out.bias),
        &mut scratch.projected,
    )?;
    for (value, &residual) in hidden.iter_mut().zip(&scratch.projected) {
        *value += residual;
    }

    compute.layer_norm_f32(
        hidden,
        &mut scratch.norm,
        rows,
        columns,
        &block.final_norm_weight,
        &block.final_norm_bias,
        config.audio.layer_norm_eps,
    )?;
    compute.gemm_f32(
        rows,
        ffn,
        columns,
        &scratch.norm,
        &block.fc1.weight_t,
        Some(&block.fc1.bias),
        &mut scratch.ffn,
    )?;
    compute.gelu_f32(&scratch.ffn, &mut scratch.activated)?;
    compute.gemm_f32(
        rows,
        columns,
        ffn,
        &scratch.activated,
        &block.fc2.weight_t,
        Some(&block.fc2.bias),
        &mut scratch.residual,
    )?;
    for (value, &residual) in hidden.iter_mut().zip(&scratch.residual) {
        *value += residual;
    }
    reject_non_finite(label, "audio encoder layer", hidden)
}

fn segmented_attention(
    compute: &Compute,
    segments: &[usize],
    config: Qwen3AsrConfig,
    scratch: &mut AudioStepScratch,
    label: &str,
) -> Result<()> {
    let columns = config.audio.d_model as usize;
    let heads = config.audio.n_head as usize;
    let head_dim = columns / heads;
    let rows = scratch.q.len() / columns;
    if segments.iter().sum::<usize>() != rows || head_dim * heads != columns {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: segmented attention shape mismatch: rows={rows}, segments={segments:?}, heads={heads}, head_dim={head_dim}, columns={columns}"
        )));
    }
    scratch.attention.fill(0.0);
    let scale = (head_dim as f32).sqrt().recip();
    let mut offset = 0;
    for &length in segments {
        resize_zero(&mut scratch.query, length * head_dim);
        resize_zero(&mut scratch.key_t, head_dim * length);
        resize_zero(&mut scratch.value, length * head_dim);
        resize_zero(&mut scratch.scores, length * length);
        resize_zero(&mut scratch.probabilities, length * length);
        resize_zero(&mut scratch.attended, length * head_dim);
        for head in 0..heads {
            for row in 0..length {
                let source = (offset + row) * columns + head * head_dim;
                let target = row * head_dim;
                scratch.query[target..target + head_dim]
                    .copy_from_slice(&scratch.q[source..source + head_dim]);
                scratch.value[target..target + head_dim]
                    .copy_from_slice(&scratch.v[source..source + head_dim]);
                for dimension in 0..head_dim {
                    scratch.key_t[dimension * length + row] = scratch.k[source + dimension];
                }
            }
            compute.gemm_f32(
                length,
                length,
                head_dim,
                &scratch.query,
                &scratch.key_t,
                None,
                &mut scratch.scores,
            )?;
            for score in &mut scratch.scores {
                *score *= scale;
            }
            compute.softmax_f32(&scratch.scores, &mut scratch.probabilities, length, length)?;
            compute.gemm_f32(
                length,
                head_dim,
                length,
                &scratch.probabilities,
                &scratch.value,
                None,
                &mut scratch.attended,
            )?;
            for row in 0..length {
                let target = (offset + row) * columns + head * head_dim;
                scratch.attention[target..target + head_dim]
                    .copy_from_slice(&scratch.attended[row * head_dim..(row + 1) * head_dim]);
            }
        }
        offset += length;
    }
    Ok(())
}

fn materialize_audio_post(
    mapped: &Qwen3AsrMappedDescriptors,
    config: Qwen3AsrConfig,
    post: &mut AudioPostBlock,
) -> Result<()> {
    let descriptors = mapped.audio_post();
    let hidden = config.audio.d_model as usize;
    let output = config.audio.output_dim as usize;
    widen_tensor(mapped, descriptors.norm_weight, &mut post.norm_weight)?;
    widen_tensor(mapped, descriptors.norm_bias, &mut post.norm_bias)?;
    materialize_linear(
        mapped,
        descriptors.proj1_weight,
        descriptors.proj1_bias,
        hidden,
        hidden,
        &mut post.proj1,
    )?;
    materialize_linear(
        mapped,
        descriptors.proj2_weight,
        descriptors.proj2_bias,
        output,
        hidden,
        &mut post.proj2,
    )
}

fn project_audio(
    compute: &Compute,
    hidden: &[f32],
    config: Qwen3AsrConfig,
    post: &AudioPostBlock,
    scratch: &mut AudioStepScratch,
    label: &str,
) -> Result<Vec<f32>> {
    let columns = config.audio.d_model as usize;
    let output = config.audio.output_dim as usize;
    let rows = hidden.len() / columns;
    resize_zero(&mut scratch.norm, hidden.len());
    resize_zero(&mut scratch.projected, hidden.len());
    resize_zero(&mut scratch.activated, hidden.len());
    compute.layer_norm_f32(
        hidden,
        &mut scratch.norm,
        rows,
        columns,
        &post.norm_weight,
        &post.norm_bias,
        config.audio.layer_norm_eps,
    )?;
    compute.gemm_f32(
        rows,
        columns,
        columns,
        &scratch.norm,
        &post.proj1.weight_t,
        Some(&post.proj1.bias),
        &mut scratch.projected,
    )?;
    compute.gelu_f32(&scratch.projected, &mut scratch.activated)?;
    let mut values = vec![0.0; rows * output];
    compute.gemm_f32(
        rows,
        output,
        columns,
        &scratch.activated,
        &post.proj2.weight_t,
        Some(&post.proj2.bias),
        &mut values,
    )?;
    reject_non_finite(label, "projected audio embeddings", &values)?;
    Ok(values)
}

fn resize_zero(values: &mut Vec<f32>, len: usize) {
    values.clear();
    values.resize(len, 0.0);
}

fn widen_tensor(
    mapped: &Qwen3AsrMappedDescriptors,
    info: &GgufTensorInfo,
    output: &mut Vec<f32>,
) -> Result<()> {
    widen_into(
        mapped.file().tensor_bytes(info),
        info.dtype,
        output,
        mapped.mapped_model(),
    )
}

fn transpose_tensor(
    mapped: &Qwen3AsrMappedDescriptors,
    info: &GgufTensorInfo,
    rows: usize,
    columns: usize,
    output: &mut Vec<f32>,
) -> Result<()> {
    transpose_widen(
        mapped.file().tensor_bytes(info),
        info.dtype,
        rows,
        columns,
        output,
        mapped.mapped_model(),
    )
}

fn reject_non_finite(label: &str, value_label: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::ModelLoad(format!(
            "{label}: {value_label} contains non-finite value {value} at index {index}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoid_uses_half_split_sin_then_cos_layout() {
        let mut values = vec![0.0; 2 * 4];
        add_sinusoidal_positions(&mut values, 2, 4, "test").expect("positions");
        assert_eq!(&values[..4], &[0.0, 0.0, 1.0, 1.0]);
        assert!((values[4] - 1.0_f32.sin()).abs() < 1.0e-6);
        assert!((values[6] - 1.0_f32.cos()).abs() < 1.0e-6);
    }

    #[test]
    fn stride2_im2col_indexing_pins_padding_edges() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let linear = DenseLinear {
            weight_t: vec![1.0; 9],
            bias: vec![0.0],
        };
        let compute = Compute::cpu();
        let (output, height, width) =
            conv2d_stride2_gelu(&compute, &input, 1, 2, 2, 1, &linear, "test").expect("conv");
        assert_eq!((height, width), (1, 1));
        assert!(output[0].is_finite());
        assert!(output[0] > 9.9);
    }
}
