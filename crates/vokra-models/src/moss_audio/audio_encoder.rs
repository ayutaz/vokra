//! Native bounded-memory MOSS-Audio tower and GatedMLP adapters.
//!
//! Conv2D is lowered to im2col + GEMM. Every learned convolution, attention,
//! normalization, FFN and adapter operation runs through one preflighted
//! [`Compute`] backend. Padding, layout transforms, residual additions,
//! sinusoidal positions and chunk assembly are deterministic host glue.

use std::sync::Mutex;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufTensorInfo;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::mapped_weights::{lock_scratch, transpose_widen, widen_into};

use super::frontend::{self, AUDIO_CHUNK_FRAMES, MossAudioFeatures, N_MELS};
use super::weights::{MossAudioAffineDescriptors, MossAudioMappedDescriptors};
use super::{MossAudioCheckpoint, MossAudioConfig};

/// Complete learned-op set of the audio tower plus four GatedMLP adapters.
pub const MOSS_AUDIO_ENCODER_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Silu,
];

/// Text-width audio embeddings and three DeepStack injection tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct MossAudioEmbeddings {
    values: Vec<f32>,
    deepstack: [Vec<f32>; 3],
    frames: usize,
    hidden_size: usize,
}

impl MossAudioEmbeddings {
    /// Primary row-major `[frames, hidden_size]` audio replacement values.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Three row-major injection tensors added after the first three Qwen3
    /// decoder layers.
    #[must_use]
    pub fn deepstack_values(&self) -> [&[f32]; 3] {
        [&self.deepstack[0], &self.deepstack[1], &self.deepstack[2]]
    }

    /// Returns the number of encoded audio frames.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    /// Returns the hidden width of every encoded frame.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[derive(Default)]
pub(super) struct MossAudioEncoderRuntime {
    convolution: Mutex<ConvolutionBank>,
    layer: Mutex<AudioLayerBlock>,
    post_norm: Mutex<AudioNormBlock>,
    adapter: Mutex<AdapterBlock>,
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
    stem_projection: DenseLinear,
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
struct AudioNormBlock {
    ready: bool,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Default)]
struct AdapterBlock {
    gate_weight_t: Vec<f32>,
    up_weight_t: Vec<f32>,
    down_weight_t: Vec<f32>,
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
    checkpoint: &MossAudioCheckpoint,
    backend: BackendKind,
    runtime: &MossAudioEncoderRuntime,
    pcm: &[f32],
) -> Result<MossAudioEmbeddings> {
    let mapped = checkpoint.mapped()?;
    let config = mapped.config();
    let compute = Compute::for_backend(backend, MOSS_AUDIO_ENCODER_HOT_OPS)?;
    let features = frontend::extract(pcm)?;

    let mut convolution = lock_scratch(&runtime.convolution, mapped.mapped_model())?;
    if !convolution.ready {
        materialize_convolution(mapped, config, &mut convolution)?;
        convolution.ready = true;
    }
    let (mut hidden, segments) = convolutional_stem(&compute, &features, config, &convolution)?;
    drop(convolution);

    let mut deepstack_audio = [Vec::new(), Vec::new(), Vec::new()];
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
        )?;
        if let Some(capture) = config
            .audio
            .deepstack_layer_indexes
            .iter()
            .position(|&index| index as usize == layer)
        {
            deepstack_audio[capture].clone_from(&hidden);
        }
    }
    drop(layer_block);
    if deepstack_audio.iter().any(Vec::is_empty) {
        return Err(VokraError::ModelLoad(
            "moss_audio: one or more configured DeepStack audio layers were not captured"
                .to_owned(),
        ));
    }

    let mut norm = lock_scratch(&runtime.post_norm, mapped.mapped_model())?;
    if !norm.ready {
        let descriptors = mapped.audio_post_norm();
        widen_tensor(mapped, descriptors.weight, &mut norm.weight)?;
        widen_tensor(mapped, descriptors.bias, &mut norm.bias)?;
        norm.ready = true;
    }
    let rows = hidden.len() / config.audio.d_model as usize;
    resize_zero(&mut scratch.norm, hidden.len());
    compute.layer_norm_f32(
        &hidden,
        &mut scratch.norm,
        rows,
        config.audio.d_model as usize,
        &norm.weight,
        &norm.bias,
        config.audio.layer_norm_eps,
    )?;
    hidden.clone_from(&scratch.norm);
    drop(norm);

    let mut adapter = lock_scratch(&runtime.adapter, mapped.mapped_model())?;
    let values = project_adapter(
        mapped,
        &compute,
        config,
        0,
        &hidden,
        &mut adapter,
        &mut scratch,
    )?;
    let mut deepstack = [Vec::new(), Vec::new(), Vec::new()];
    for (index, source) in deepstack_audio.iter().enumerate() {
        deepstack[index] = project_adapter(
            mapped,
            &compute,
            config,
            index + 1,
            source,
            &mut adapter,
            &mut scratch,
        )?;
    }
    let hidden_size = config.text.hidden_size as usize;
    if values.len() != rows * hidden_size
        || deepstack
            .iter()
            .any(|values| values.len() != rows * hidden_size)
    {
        return Err(VokraError::ModelLoad(
            "moss_audio: adapter output shape disagrees with the authenticated text width"
                .to_owned(),
        ));
    }
    Ok(MossAudioEmbeddings {
        values,
        deepstack,
        frames: rows,
        hidden_size,
    })
}

fn materialize_convolution(
    mapped: &MossAudioMappedDescriptors,
    config: MossAudioConfig,
    bank: &mut ConvolutionBank,
) -> Result<()> {
    let channels = config.audio.downsample_hidden_size as usize;
    materialize_affine(
        mapped,
        mapped.convolution(1),
        channels,
        3 * 3,
        &mut bank.conv1,
    )?;
    materialize_affine(
        mapped,
        mapped.convolution(2),
        channels,
        channels * 3 * 3,
        &mut bank.conv2,
    )?;
    materialize_affine(
        mapped,
        mapped.convolution(3),
        channels,
        channels * 3 * 3,
        &mut bank.conv3,
    )?;
    materialize_affine(
        mapped,
        mapped.stem_projection(),
        config.audio.d_model as usize,
        channels * 16,
        &mut bank.stem_projection,
    )
}

fn materialize_affine(
    mapped: &MossAudioMappedDescriptors,
    descriptors: MossAudioAffineDescriptors<'_>,
    output: usize,
    input: usize,
    linear: &mut DenseLinear,
) -> Result<()> {
    transpose_tensor(
        mapped,
        descriptors.weight,
        output,
        input,
        &mut linear.weight_t,
    )?;
    match descriptors.bias {
        Some(info) => widen_tensor(mapped, info, &mut linear.bias),
        None => {
            linear.bias.clear();
            Ok(())
        }
    }
}

fn convolutional_stem(
    compute: &Compute,
    features: &MossAudioFeatures,
    config: MossAudioConfig,
    bank: &ConvolutionBank,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let frames = features.frames;
    let chunks = frames.div_ceil(AUDIO_CHUNK_FRAMES);
    let padded_width = frames.min(AUDIO_CHUNK_FRAMES);
    let padded_after_cnn = frontend::encoded_frames(padded_width);
    let d_model = config.audio.d_model as usize;
    let channels = config.audio.downsample_hidden_size as usize;
    let expected_rows = (0..chunks)
        .map(|chunk| {
            frontend::encoded_frames((frames - chunk * AUDIO_CHUNK_FRAMES).min(AUDIO_CHUNK_FRAMES))
        })
        .sum::<usize>();
    let mut hidden = Vec::with_capacity(expected_rows * d_model);
    let mut segments = Vec::with_capacity(chunks);

    for chunk in 0..chunks {
        let frame_start = chunk * AUDIO_CHUNK_FRAMES;
        let actual_width = (frames - frame_start).min(AUDIO_CHUNK_FRAMES);
        let actual_after_cnn = frontend::encoded_frames(actual_width);
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
        )?;
        let (stage2, height2, width2) = conv2d_stride2_gelu(
            compute,
            &stage1,
            channels,
            height1,
            width1,
            channels,
            &bank.conv2,
        )?;
        let (stage3, height3, width3) = conv2d_stride2_gelu(
            compute,
            &stage2,
            channels,
            height2,
            width2,
            channels,
            &bank.conv3,
        )?;
        if height3 != 16 || width3 != padded_after_cnn {
            return Err(VokraError::ModelLoad(format!(
                "moss_audio: convolutional stem produced [C,{height3},{width3}], expected [C,16,{padded_after_cnn}]"
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
            &bank.stem_projection.weight_t,
            Some(&bank.stem_projection.bias),
            &mut embedded,
        )?;
        add_sinusoidal_positions(&mut embedded, width3, d_model)?;
        hidden.extend_from_slice(&embedded[..actual_after_cnn * d_model]);
        segments.push(actual_after_cnn);
    }

    if hidden.len() != expected_rows * d_model {
        return Err(VokraError::ModelLoad(format!(
            "moss_audio: convolutional stem retained {} rows, expected {expected_rows}",
            hidden.len() / d_model
        )));
    }
    reject_non_finite("convolutional embeddings", &hidden)?;
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
) -> Result<(Vec<f32>, usize, usize)> {
    let expected = in_channels
        .checked_mul(input_height)
        .and_then(|value| value.checked_mul(input_width))
        .ok_or_else(|| {
            VokraError::InvalidArgument("moss_audio: conv input size overflow".into())
        })?;
    if input.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "moss_audio: conv input has {} values, expected {expected}",
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

fn add_sinusoidal_positions(values: &mut [f32], rows: usize, columns: usize) -> Result<()> {
    if columns < 4 || !columns.is_multiple_of(2) || values.len() != rows * columns {
        return Err(VokraError::InvalidArgument(format!(
            "moss_audio: sinusoidal position shape mismatch: values={}, rows={rows}, columns={columns}",
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
    mapped: &MossAudioMappedDescriptors,
    layer: usize,
    config: MossAudioConfig,
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
        Some(descriptors.q_bias),
        hidden,
        hidden,
        &mut block.q,
    )?;
    materialize_linear(
        mapped,
        descriptors.k_weight,
        None,
        hidden,
        hidden,
        &mut block.k,
    )?;
    materialize_linear(
        mapped,
        descriptors.v_weight,
        Some(descriptors.v_bias),
        hidden,
        hidden,
        &mut block.v,
    )?;
    materialize_linear(
        mapped,
        descriptors.out_weight,
        Some(descriptors.out_bias),
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
        Some(descriptors.fc1_bias),
        ffn,
        hidden,
        &mut block.fc1,
    )?;
    materialize_linear(
        mapped,
        descriptors.fc2_weight,
        Some(descriptors.fc2_bias),
        hidden,
        ffn,
        &mut block.fc2,
    )
}

fn materialize_linear(
    mapped: &MossAudioMappedDescriptors,
    weight: &GgufTensorInfo,
    bias: Option<&GgufTensorInfo>,
    output: usize,
    input: usize,
    linear: &mut DenseLinear,
) -> Result<()> {
    transpose_tensor(mapped, weight, output, input, &mut linear.weight_t)?;
    match bias {
        Some(info) => widen_tensor(mapped, info, &mut linear.bias),
        None => {
            linear.bias.clear();
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encoder_layer(
    compute: &Compute,
    hidden: &mut [f32],
    segments: &[usize],
    config: MossAudioConfig,
    block: &AudioLayerBlock,
    scratch: &mut AudioStepScratch,
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
        None,
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
    segmented_attention(compute, segments, config, scratch)?;
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
    reject_non_finite("audio encoder layer", hidden)
}

fn segmented_attention(
    compute: &Compute,
    segments: &[usize],
    config: MossAudioConfig,
    scratch: &mut AudioStepScratch,
) -> Result<()> {
    let columns = config.audio.d_model as usize;
    let heads = config.audio.n_head as usize;
    let head_dim = columns / heads;
    let rows = scratch.q.len() / columns;
    if segments.iter().sum::<usize>() != rows || head_dim * heads != columns {
        return Err(VokraError::InvalidArgument(format!(
            "moss_audio: segmented attention shape mismatch: rows={rows}, segments={segments:?}, heads={heads}, head_dim={head_dim}, columns={columns}"
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

#[allow(clippy::too_many_arguments)]
fn project_adapter(
    mapped: &MossAudioMappedDescriptors,
    compute: &Compute,
    config: MossAudioConfig,
    index: usize,
    input: &[f32],
    block: &mut AdapterBlock,
    scratch: &mut AudioStepScratch,
) -> Result<Vec<f32>> {
    let descriptors = mapped.adapter(index);
    let rows = input.len() / config.audio.output_dim as usize;
    let input_width = config.audio.output_dim as usize;
    let adapter_width = config.adapter_hidden_size as usize;
    let output_width = config.text.hidden_size as usize;
    transpose_tensor(
        mapped,
        descriptors.gate,
        adapter_width,
        input_width,
        &mut block.gate_weight_t,
    )?;
    transpose_tensor(
        mapped,
        descriptors.up,
        adapter_width,
        input_width,
        &mut block.up_weight_t,
    )?;
    transpose_tensor(
        mapped,
        descriptors.down,
        output_width,
        adapter_width,
        &mut block.down_weight_t,
    )?;
    resize_zero(&mut scratch.ffn, rows * adapter_width);
    resize_zero(&mut scratch.activated, rows * adapter_width);
    resize_zero(&mut scratch.residual, rows * adapter_width);
    compute.gemm_f32(
        rows,
        adapter_width,
        input_width,
        input,
        &block.gate_weight_t,
        None,
        &mut scratch.ffn,
    )?;
    compute.silu_f32(&scratch.ffn, &mut scratch.activated)?;
    compute.gemm_f32(
        rows,
        adapter_width,
        input_width,
        input,
        &block.up_weight_t,
        None,
        &mut scratch.residual,
    )?;
    for (gate, up) in scratch.activated.iter_mut().zip(&scratch.residual) {
        *gate *= up;
    }
    let mut output = vec![0.0; rows * output_width];
    compute.gemm_f32(
        rows,
        output_width,
        adapter_width,
        &scratch.activated,
        &block.down_weight_t,
        None,
        &mut output,
    )?;
    reject_non_finite("GatedMLP adapter output", &output)?;
    Ok(output)
}

fn resize_zero(values: &mut Vec<f32>, len: usize) {
    values.clear();
    values.resize(len, 0.0);
}

fn widen_tensor(
    mapped: &MossAudioMappedDescriptors,
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
    mapped: &MossAudioMappedDescriptors,
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

fn reject_non_finite(value_label: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::ModelLoad(format!(
            "moss_audio: {value_label} contains non-finite value {value} at index {index}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoid_uses_half_split_sin_then_cos_layout() {
        let mut values = vec![0.0; 8];
        add_sinusoidal_positions(&mut values, 2, 4).expect("positions");
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
            conv2d_stride2_gelu(&compute, &input, 1, 2, 2, 1, &linear).expect("conv");
        assert_eq!((height, width), (1, 1));
        assert!(output[0].is_finite());
        assert!(output[0] > 9.9);
    }
}
