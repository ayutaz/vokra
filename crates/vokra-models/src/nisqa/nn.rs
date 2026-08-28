use vokra_core::ir::graph::{
    MelAttrs, MelInterp, MelNorm, MelScale, Normalization, PadMode, StftAttrs, Window,
    WindowSymmetry,
};
use vokra_core::{Result, VokraError};
use vokra_ops::{mel_filterbank, stft};

use crate::compute::Compute;

use super::weights::{BatchNorm, Conv2d, Linear, NisqaWeights, Norm, PoolHead};
use super::{MEL_DB_AMIN, MEL_DB_TOP_DB, NisqaFrontEndSpec, NisqaScore, NisqaTopologySpec};

const BATCH_NORM_EPS: f32 = 1.0e-5;
const LAYER_NORM_EPS: f32 = 1.0e-5;
const IM2COL_CHUNK: usize = 4_096;
const MODEL_WIDTH: usize = 64;

#[derive(Debug)]
struct Tensor4 {
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
        let expected = checked_product("Tensor4", &[batch, channels, height, width])?;
        if data.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "nisqa: Tensor4 has {} values, expected {batch}x{channels}x{height}x{width}={expected}",
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

    fn index(&self, batch: usize, channel: usize, y: usize, x: usize) -> usize {
        (((batch * self.channels + channel) * self.height + y) * self.width) + x
    }
}

pub(super) fn score(
    compute: &Compute,
    weights: &NisqaWeights,
    front_end: &NisqaFrontEndSpec,
    topology: &NisqaTopologySpec,
    pcm: &[f32],
    sample_rate: u32,
) -> Result<NisqaScore> {
    validate_pcm(pcm, sample_rate, front_end)?;
    if topology.td_sa_nhead != 1 {
        return Err(VokraError::ModelLoad(format!(
            "nisqa: exact public checkpoint requires one attention head, got {}",
            topology.td_sa_nhead
        )));
    }
    let mut value = frontend(pcm, sample_rate, front_end)?;
    for (index, block) in weights.conv.iter().enumerate() {
        let (pad_h, pad_w) = if index == 5 { (1, 0) } else { (1, 1) };
        value = conv2d(compute, &value, &block.conv, pad_h, pad_w)?;
        batch_norm(&mut value, &block.norm)?;
        relu(&mut value.data);
        match index {
            0 => {
                value = adaptive_max_pool(
                    &value,
                    topology.cnn_pool[0][0] as usize,
                    topology.cnn_pool[0][1] as usize,
                )?;
            }
            1 => {
                value = adaptive_max_pool(
                    &value,
                    topology.cnn_pool[1][0] as usize,
                    topology.cnn_pool[1][1] as usize,
                )?;
            }
            3 => {
                value = adaptive_max_pool(
                    &value,
                    topology.cnn_pool[2][0] as usize,
                    topology.cnn_pool[2][1] as usize,
                )?;
            }
            _ => {}
        }
    }

    let expected_height = topology.cnn_pool[2][0] as usize;
    if value.channels != MODEL_WIDTH || value.height != expected_height || value.width != 1 {
        return Err(VokraError::ModelLoad(format!(
            "nisqa: AdaptCNN output is [{},{},{},{}], expected [segments,{MODEL_WIDTH},{expected_height},1]",
            value.batch, value.channels, value.height, value.width
        )));
    }
    let cnn_width = checked_product("CNN flatten", &[value.channels, value.height])?;
    if cnn_width != weights.input_linear.input {
        return Err(VokraError::ModelLoad(format!(
            "nisqa: AdaptCNN flatten width {cnn_width}, expected {}",
            weights.input_linear.input
        )));
    }

    let mut sequence = linear(compute, &value.data, &weights.input_linear)?;
    sequence = layer_norm(compute, &sequence, value.batch, &weights.input_norm)?;
    for layer in &weights.attention {
        sequence = attention_layer(compute, &sequence, value.batch, layer)?;
    }

    let mut heads = Vec::with_capacity(weights.pool_heads.len());
    for head in &weights.pool_heads {
        heads.push(pool_head(compute, &sequence, value.batch, head)?);
    }
    NisqaScore::from_heads(&heads)
}

fn validate_pcm(pcm: &[f32], sample_rate: u32, front_end: &NisqaFrontEndSpec) -> Result<()> {
    if pcm.is_empty() {
        return Err(VokraError::InvalidArgument(
            "nisqa: PCM must contain at least one mono sample".to_owned(),
        ));
    }
    if sample_rate == 0 {
        return Err(VokraError::InvalidArgument(
            "nisqa: sample_rate must be positive".to_owned(),
        ));
    }
    if front_end.sample_rate != 0 && front_end.sample_rate != sample_rate {
        return Err(VokraError::InvalidArgument(format!(
            "nisqa: checkpoint requires {} Hz PCM, got {sample_rate} Hz; resample explicitly before scoring",
            front_end.sample_rate
        )));
    }
    if let Some((index, _)) = pcm.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "nisqa: PCM sample {index} is not finite"
        )));
    }
    Ok(())
}

fn frontend(pcm: &[f32], sample_rate: u32, spec: &NisqaFrontEndSpec) -> Result<Tensor4> {
    let n_fft = spec.n_fft as usize;
    // Upstream uses `int(seconds * sr)`, i.e. truncation for these positive values.
    let hop_length = (spec.hop_length_sec * sample_rate as f32) as usize;
    let win_length = (spec.win_length_sec * sample_rate as f32) as usize;
    if hop_length == 0 || win_length == 0 || win_length > n_fft {
        return Err(VokraError::InvalidArgument(format!(
            "nisqa: invalid native-rate STFT at {sample_rate} Hz: n_fft={n_fft}, hop={hop_length}, win={win_length}"
        )));
    }
    let spectrogram = stft(
        pcm,
        &StftAttrs {
            n_fft,
            hop_length,
            win_length,
            window: Window::Hann,
            window_symmetry: WindowSymmetry::Periodic,
            center: true,
            pad_mode: PadMode::Reflect,
            normalization: Normalization::Backward,
            causal: false,
            real_input: true,
        },
    )?;
    let n_mels = spec.n_mels as usize;
    let mut mel = mel_filterbank(&MelAttrs {
        sample_rate,
        n_fft,
        n_mels,
        fmin: 0.0,
        fmax: Some(spec.fmax),
        scale: MelScale::Slaney,
        norm: MelNorm::Slaney,
        interp: MelInterp::Hz,
    })
    .apply(&spectrogram.magnitude(), spectrogram.frames);
    amplitude_to_db(&mut mel);

    let seg_length = spec.seg_length as usize;
    if spectrogram.frames < seg_length {
        return Err(VokraError::InvalidArgument(format!(
            "nisqa: mel spectrogram has {} frames, fewer than segment length {seg_length}",
            spectrogram.frames
        )));
    }
    let available = spectrogram.frames - (seg_length - 1);
    let seg_hop = spec.seg_hop_length as usize;
    let segments = available.div_ceil(seg_hop);
    if segments > spec.max_segments as usize {
        return Err(VokraError::InvalidArgument(format!(
            "nisqa: clip produces {segments} segments, exceeding checkpoint maximum {}",
            spec.max_segments
        )));
    }
    let mut output = vec![0.0f32; segments * n_mels * seg_length];
    for segment in 0..segments {
        let start = segment * seg_hop;
        for mel_bin in 0..n_mels {
            for offset in 0..seg_length {
                output[(segment * n_mels + mel_bin) * seg_length + offset] =
                    mel[(start + offset) * n_mels + mel_bin];
            }
        }
    }
    Tensor4::new(output, segments, 1, n_mels, seg_length)
}

fn amplitude_to_db(values: &mut [f32]) {
    let mut maximum = f32::NEG_INFINITY;
    for value in values.iter_mut() {
        *value = 20.0 * value.max(MEL_DB_AMIN).log10();
        maximum = maximum.max(*value);
    }
    let floor = maximum - MEL_DB_TOP_DB;
    for value in values {
        *value = value.max(floor);
    }
}

fn conv2d(
    compute: &Compute,
    input: &Tensor4,
    weights: &Conv2d,
    pad_h: usize,
    pad_w: usize,
) -> Result<Tensor4> {
    if input.channels != weights.input
        || weights.weight.len()
            != weights.output * weights.input * weights.kernel_h * weights.kernel_w
        || weights.bias.len() != weights.output
    {
        return Err(VokraError::ModelLoad(format!(
            "nisqa: Conv2d shape mismatch (activation channels {}, weight {}->{}, kernel {}x{})",
            input.channels, weights.input, weights.output, weights.kernel_h, weights.kernel_w
        )));
    }
    let padded_h = input.height.checked_add(2 * pad_h).ok_or_else(|| {
        VokraError::InvalidArgument("nisqa: Conv2d padded height overflow".to_owned())
    })?;
    let padded_w = input.width.checked_add(2 * pad_w).ok_or_else(|| {
        VokraError::InvalidArgument("nisqa: Conv2d padded width overflow".to_owned())
    })?;
    if padded_h < weights.kernel_h || padded_w < weights.kernel_w {
        return Err(VokraError::InvalidArgument(
            "nisqa: Conv2d kernel exceeds padded activation".to_owned(),
        ));
    }
    let output_h = padded_h - weights.kernel_h + 1;
    let output_w = padded_w - weights.kernel_w + 1;
    let spatial = checked_product("Conv2d spatial", &[output_h, output_w])?;
    let total_positions = checked_product("Conv2d positions", &[input.batch, spatial])?;
    let patch = checked_product(
        "Conv2d patch",
        &[weights.input, weights.kernel_h, weights.kernel_w],
    )?;
    let mut output = vec![0.0f32; input.batch * weights.output * spatial];
    for chunk_start in (0..total_positions).step_by(IM2COL_CHUNK) {
        let chunk = (total_positions - chunk_start).min(IM2COL_CHUNK);
        let mut columns = vec![0.0f32; patch * chunk];
        for input_channel in 0..weights.input {
            for kernel_y in 0..weights.kernel_h {
                for kernel_x in 0..weights.kernel_w {
                    let patch_index =
                        (input_channel * weights.kernel_h + kernel_y) * weights.kernel_w + kernel_x;
                    for local in 0..chunk {
                        let global = chunk_start + local;
                        let batch = global / spatial;
                        let position = global % spatial;
                        let output_y = position / output_w;
                        let output_x = position % output_w;
                        let source_y = output_y + kernel_y;
                        let source_x = output_x + kernel_x;
                        if source_y >= pad_h
                            && source_x >= pad_w
                            && source_y - pad_h < input.height
                            && source_x - pad_w < input.width
                        {
                            columns[patch_index * chunk + local] = input.data[input.index(
                                batch,
                                input_channel,
                                source_y - pad_h,
                                source_x - pad_w,
                            )];
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
        for channel in 0..weights.output {
            for local in 0..chunk {
                let global = chunk_start + local;
                let batch = global / spatial;
                let position = global % spatial;
                output[(batch * weights.output + channel) * spatial + position] =
                    block[channel * chunk + local] + weights.bias[channel];
            }
        }
    }
    Tensor4::new(output, input.batch, weights.output, output_h, output_w)
}

fn batch_norm(value: &mut Tensor4, norm: &BatchNorm) -> Result<()> {
    if norm.gamma.len() != value.channels
        || norm.beta.len() != value.channels
        || norm.running_mean.len() != value.channels
        || norm.running_var.len() != value.channels
    {
        return Err(VokraError::ModelLoad(format!(
            "nisqa: BatchNorm width mismatch for {} channels",
            value.channels
        )));
    }
    let spatial = value.height * value.width;
    for batch in 0..value.batch {
        for channel in 0..value.channels {
            let scale = norm.gamma[channel] / (norm.running_var[channel] + BATCH_NORM_EPS).sqrt();
            let shift = norm.beta[channel] - norm.running_mean[channel] * scale;
            let start = (batch * value.channels + channel) * spatial;
            for item in &mut value.data[start..start + spatial] {
                *item = *item * scale + shift;
            }
        }
    }
    Ok(())
}

fn adaptive_max_pool(input: &Tensor4, output_h: usize, output_w: usize) -> Result<Tensor4> {
    if output_h == 0 || output_w == 0 {
        return Err(VokraError::InvalidArgument(
            "nisqa: adaptive max pool output must be non-zero".to_owned(),
        ));
    }
    let mut output = vec![f32::NEG_INFINITY; input.batch * input.channels * output_h * output_w];
    for batch in 0..input.batch {
        for channel in 0..input.channels {
            for output_y in 0..output_h {
                let start_y = output_y * input.height / output_h;
                let end_y = ((output_y + 1) * input.height).div_ceil(output_h);
                for output_x in 0..output_w {
                    let start_x = output_x * input.width / output_w;
                    let end_x = ((output_x + 1) * input.width).div_ceil(output_w);
                    let mut maximum = f32::NEG_INFINITY;
                    for y in start_y..end_y {
                        for x in start_x..end_x {
                            maximum = maximum.max(input.data[input.index(batch, channel, y, x)]);
                        }
                    }
                    output[(((batch * input.channels + channel) * output_h + output_y)
                        * output_w)
                        + output_x] = maximum;
                }
            }
        }
    }
    Tensor4::new(output, input.batch, input.channels, output_h, output_w)
}

fn attention_layer(
    compute: &Compute,
    input: &[f32],
    rows: usize,
    layer: &super::weights::AttentionLayer,
) -> Result<Vec<f32>> {
    let qkv = linear(compute, input, &layer.in_proj)?;
    let mut query = vec![0.0f32; rows * MODEL_WIDTH];
    let mut key = vec![0.0f32; rows * MODEL_WIDTH];
    let mut value = vec![0.0f32; rows * MODEL_WIDTH];
    for row in 0..rows {
        let source = &qkv[row * 3 * MODEL_WIDTH..(row + 1) * 3 * MODEL_WIDTH];
        query[row * MODEL_WIDTH..(row + 1) * MODEL_WIDTH].copy_from_slice(&source[..MODEL_WIDTH]);
        key[row * MODEL_WIDTH..(row + 1) * MODEL_WIDTH]
            .copy_from_slice(&source[MODEL_WIDTH..2 * MODEL_WIDTH]);
        value[row * MODEL_WIDTH..(row + 1) * MODEL_WIDTH]
            .copy_from_slice(&source[2 * MODEL_WIDTH..]);
    }
    let mut key_transposed = vec![0.0f32; MODEL_WIDTH * rows];
    for row in 0..rows {
        for column in 0..MODEL_WIDTH {
            key_transposed[column * rows + row] = key[row * MODEL_WIDTH + column];
        }
    }
    let mut logits = vec![0.0f32; rows * rows];
    compute.gemm_f32(
        rows,
        rows,
        MODEL_WIDTH,
        &query,
        &key_transposed,
        None,
        &mut logits,
    )?;
    let scale = 1.0f32 / (MODEL_WIDTH as f32).sqrt();
    for item in &mut logits {
        *item *= scale;
    }
    let mut probabilities = vec![0.0f32; logits.len()];
    compute.softmax_f32(&logits, &mut probabilities, rows, rows)?;
    let mut context = vec![0.0f32; rows * MODEL_WIDTH];
    compute.gemm_f32(
        rows,
        MODEL_WIDTH,
        rows,
        &probabilities,
        &value,
        None,
        &mut context,
    )?;
    let projected = linear(compute, &context, &layer.out_proj)?;
    let residual: Vec<f32> = input
        .iter()
        .zip(projected)
        .map(|(&left, right)| left + right)
        .collect();
    let normalized = layer_norm(compute, &residual, rows, &layer.norm1)?;
    let mut hidden = linear(compute, &normalized, &layer.linear1)?;
    relu(&mut hidden);
    let feed_forward = linear(compute, &hidden, &layer.linear2)?;
    let residual: Vec<f32> = normalized
        .iter()
        .zip(feed_forward)
        .map(|(&left, right)| left + right)
        .collect();
    layer_norm(compute, &residual, rows, &layer.norm2)
}

fn pool_head(compute: &Compute, input: &[f32], rows: usize, head: &PoolHead) -> Result<f32> {
    let mut hidden = linear(compute, input, &head.linear1)?;
    relu(&mut hidden);
    let logits = linear(compute, &hidden, &head.linear2)?;
    let mut attention = vec![0.0f32; rows];
    compute.softmax_f32(&logits, &mut attention, 1, rows)?;
    let mut pooled = vec![0.0f32; MODEL_WIDTH];
    for row in 0..rows {
        for column in 0..MODEL_WIDTH {
            pooled[column] += attention[row] * input[row * MODEL_WIDTH + column];
        }
    }
    Ok(linear(compute, &pooled, &head.linear3)?[0])
}

fn linear(compute: &Compute, input: &[f32], layer: &Linear) -> Result<Vec<f32>> {
    if input.is_empty()
        || input.len() % layer.input != 0
        || layer.weight_io.len() != layer.input * layer.output
        || layer.bias.len() != layer.output
    {
        return Err(VokraError::ModelLoad(format!(
            "nisqa: Linear shape mismatch input={}, expected width {}, output {}",
            input.len(),
            layer.input,
            layer.output
        )));
    }
    let rows = input.len() / layer.input;
    let mut output = vec![0.0f32; rows * layer.output];
    compute.gemm_f32(
        rows,
        layer.output,
        layer.input,
        input,
        &layer.weight_io,
        Some(&layer.bias),
        &mut output,
    )?;
    Ok(output)
}

fn layer_norm(compute: &Compute, input: &[f32], rows: usize, norm: &Norm) -> Result<Vec<f32>> {
    if norm.gamma.len() != MODEL_WIDTH || norm.beta.len() != MODEL_WIDTH {
        return Err(VokraError::ModelLoad(
            "nisqa: LayerNorm parameter width mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0f32; input.len()];
    compute.layer_norm_f32(
        input,
        &mut output,
        rows,
        MODEL_WIDTH,
        &norm.gamma,
        &norm.beta,
        LAYER_NORM_EPS,
    )?;
    Ok(output)
}

fn relu(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

fn checked_product(label: &str, dimensions: &[usize]) -> Result<usize> {
    dimensions.iter().try_fold(1usize, |product, &dimension| {
        product.checked_mul(dimension).ok_or_else(|| {
            VokraError::InvalidArgument(format!("nisqa: {label} shape product overflow"))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_pool_uses_pytorch_floor_ceil_bins() {
        let input =
            Tensor4::new((0..15).map(|value| value as f32).collect(), 1, 1, 3, 5).expect("input");
        let output = adaptive_max_pool(&input, 2, 3).expect("pool");
        assert_eq!(output.data, [6.0, 8.0, 9.0, 11.0, 13.0, 14.0]);
    }

    #[test]
    fn amplitude_db_has_absolute_reference_and_top_db_floor() {
        let mut values = [1.0, 0.1, 1.0e-8];
        amplitude_to_db(&mut values);
        assert!((values[0] - 0.0).abs() < 1.0e-6);
        assert!((values[1] + 20.0).abs() < 1.0e-5);
        assert!((values[2] + 80.0).abs() < 1.0e-5);
    }
}
