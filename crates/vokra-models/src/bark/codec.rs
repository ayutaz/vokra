//! Bark's embedded causal 24 kHz EnCodec decoder.
//!
//! This is intentionally distinct from `audiocraft_encodec`: Bark embeds the
//! 24 kHz causal release (`num_filters=32`, ratios `[8,5,4,2]`, learned
//! residual shortcuts), while MusicGen carries a non-causal 32 kHz topology.
//! Sharing either padding or channel tables would generate plausible but
//! numerically wrong audio.

use vokra_core::{Result, VokraError};
use vokra_ops::{CodebookTable, EncodecRvqAttrs};

use crate::compute::Compute;

use super::generation::{BarkGeneratedCodes, BarkGenerationConfig};
use super::weights::BarkMappedWeights;
use super::{BarkModel, CODEBOOK_DIM, CODEBOOK_SIZE, CODEBOOKS_USED, SAMPLE_RATE};

const LABEL: &str = "bark/codec";
const NUM_FILTERS: usize = 32;
const LSTM_DIMENSION: usize = 512;
const LSTM_LAYERS: usize = 2;
const RATIOS: [usize; 4] = [8, 5, 4, 2];
const FRAME_HOP: usize = 320;

/// Bark synthesis result with explicit timebase and reusable codec codes.
#[derive(Debug, Clone, PartialEq)]
pub struct BarkSynthesis {
    /// Mono PCM at [`sample_rate`](Self::sample_rate).
    pub pcm: Vec<f32>,
    /// Output sample rate (24 kHz for both releases).
    pub sample_rate: u32,
    /// Generated frame-major eight-codebook packet.
    pub codes: BarkGeneratedCodes,
}

impl BarkModel {
    /// Decodes authenticated frame-major Bark codes to mono 24 kHz PCM.
    pub fn decode_codes(&self, codes: &BarkGeneratedCodes) -> Result<Vec<f32>> {
        if codes.frames() == 0 {
            return Err(VokraError::InvalidArgument(
                "bark/codec: frames must be > 0".to_owned(),
            ));
        }
        let weights = self.mapped()?;
        let compute = Compute::for_backend(self.backend, super::BARK_HOT_OPS)?;
        decode(weights, &compute, codes)
    }

    /// Runs token generation and the embedded EnCodec decoder end to end.
    pub fn synthesize_tokens(
        &self,
        text_token_ids: &[u32],
        attention_mask: Option<&[bool]>,
        generation: &BarkGenerationConfig,
    ) -> Result<BarkSynthesis> {
        let codes = self.generate_codes_from_tokens(text_token_ids, attention_mask, generation)?;
        let pcm = self.decode_codes(&codes)?;
        Ok(BarkSynthesis {
            pcm,
            sample_rate: SAMPLE_RATE,
            codes,
        })
    }
}

fn decode(
    weights: &BarkMappedWeights,
    compute: &Compute,
    codes: &BarkGeneratedCodes,
) -> Result<Vec<f32>> {
    let mut codebooks = Vec::with_capacity(CODEBOOKS_USED);
    for index in 0..CODEBOOKS_USED {
        let values = weights
            .tensor(
                &format!("codec_model.quantizer.layers.{index}.codebook.embed"),
                &[CODEBOOK_SIZE, CODEBOOK_DIM],
            )?
            .to_vec();
        codebooks.push(CodebookTable::new(CODEBOOK_SIZE, CODEBOOK_DIM, values)?);
    }
    let latent_time_major = compute.encodec_rvq_f32(
        codes.as_frame_major(),
        codes.frames(),
        &codebooks,
        &EncodecRvqAttrs {
            n_codebooks: CODEBOOKS_USED,
            codebook_size: CODEBOOK_SIZE,
            d_model: CODEBOOK_DIM,
        },
    )?;
    let mut latent = vec![0.0f32; latent_time_major.len()];
    for frame in 0..codes.frames() {
        for channel in 0..CODEBOOK_DIM {
            latent[channel * codes.frames() + frame] =
                latent_time_major[frame * CODEBOOK_DIM + channel];
        }
    }

    let decoder = SeanetDecoder::bind(weights)?;
    let (pcm, samples) = decoder.forward(&latent, codes.frames(), compute)?;
    let expected = codes.frames().checked_mul(FRAME_HOP).ok_or_else(|| {
        VokraError::InvalidArgument("bark/codec: frames * frame hop overflows usize".to_owned())
    })?;
    if samples != expected || pcm.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: decoder emitted {} values / {samples} samples, expected {expected}",
            pcm.len()
        )));
    }
    reject_non_finite("decoded PCM", &pcm)?;
    Ok(pcm)
}

#[derive(Debug)]
struct SeanetDecoder {
    initial: Conv1d,
    lstm: Lstm,
    upsample: Vec<ConvTranspose1d>,
    residuals: Vec<ResidualBlock>,
    final_conv: Conv1d,
}

impl SeanetDecoder {
    fn bind(weights: &BarkMappedWeights) -> Result<Self> {
        let root = "codec_model.decoder.layers";
        let initial = Conv1d::bind_weight_norm(
            weights,
            &format!("{root}.0.conv"),
            CODEBOOK_DIM,
            LSTM_DIMENSION,
            7,
        )?;
        let lstm = Lstm::bind(
            weights,
            &format!("{root}.1.lstm"),
            LSTM_DIMENSION,
            LSTM_LAYERS,
        )?;
        let mut upsample = Vec::with_capacity(RATIOS.len());
        let mut residuals = Vec::with_capacity(RATIOS.len());
        for (transpose, residual, channels, next, ratio) in [
            (3usize, 4usize, 512usize, 256usize, 8usize),
            (6, 7, 256, 128, 5),
            (9, 10, 128, 64, 4),
            (12, 13, 64, 32, 2),
        ] {
            upsample.push(ConvTranspose1d::bind_weight_norm(
                weights,
                &format!("{root}.{transpose}.conv"),
                channels,
                next,
                ratio * 2,
                ratio,
            )?);
            residuals.push(ResidualBlock::bind(
                weights,
                &format!("{root}.{residual}"),
                next,
            )?);
        }
        Ok(Self {
            initial,
            lstm,
            upsample,
            residuals,
            final_conv: Conv1d::bind_weight_norm(
                weights,
                &format!("{root}.15.conv"),
                NUM_FILTERS,
                1,
                7,
            )?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        input_len: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        let (mut hidden, mut time) = self.initial.forward(input, input_len, compute)?;
        hidden = self.lstm.forward(&hidden, time, compute)?;
        for stage in 0..RATIOS.len() {
            hidden = elu(compute, &hidden)?;
            (hidden, time) = self.upsample[stage].forward(&hidden, time, compute)?;
            hidden = self.residuals[stage].forward(&hidden, time, compute)?;
        }
        hidden = elu(compute, &hidden)?;
        self.final_conv.forward(&hidden, time, compute)
    }
}

#[derive(Debug)]
struct ResidualBlock {
    first: Conv1d,
    second: Conv1d,
    shortcut: Conv1d,
}

impl ResidualBlock {
    fn bind(weights: &BarkMappedWeights, prefix: &str, channels: usize) -> Result<Self> {
        let hidden = channels / 2;
        Ok(Self {
            first: Conv1d::bind_weight_norm(
                weights,
                &format!("{prefix}.block.1.conv"),
                channels,
                hidden,
                3,
            )?,
            second: Conv1d::bind_weight_norm(
                weights,
                &format!("{prefix}.block.3.conv"),
                hidden,
                channels,
                1,
            )?,
            shortcut: Conv1d::bind_weight_norm(
                weights,
                &format!("{prefix}.shortcut.conv"),
                channels,
                channels,
                1,
            )?,
        })
    }

    fn forward(&self, input: &[f32], time: usize, compute: &Compute) -> Result<Vec<f32>> {
        require_layout("residual input", input, self.first.input_channels, time)?;
        let (shortcut, shortcut_time) = self.shortcut.forward(input, time, compute)?;
        let hidden = elu(compute, input)?;
        let (hidden, first_time) = self.first.forward(&hidden, time, compute)?;
        let hidden = elu(compute, &hidden)?;
        let (mut hidden, second_time) = self.second.forward(&hidden, first_time, compute)?;
        if shortcut_time != time
            || first_time != time
            || second_time != time
            || hidden.len() != shortcut.len()
        {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: residual branch changed shape: shortcut={shortcut_time}, first={first_time}, second={second_time}, values={}/{}",
                hidden.len(),
                shortcut.len()
            )));
        }
        add_assign(&mut hidden, &shortcut)?;
        Ok(hidden)
    }
}

#[derive(Debug)]
struct Conv1d {
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Conv1d {
    fn bind_weight_norm(
        weights: &BarkMappedWeights,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
    ) -> Result<Self> {
        let magnitude = weights.tensor(&format!("{prefix}.weight_g"), &[output_channels, 1, 1])?;
        let direction = weights.tensor(
            &format!("{prefix}.weight_v"),
            &[output_channels, input_channels, kernel],
        )?;
        Ok(Self {
            input_channels,
            output_channels,
            kernel,
            weight: reconstruct_weight_norm(
                magnitude,
                direction,
                output_channels,
                input_channels * kernel,
                prefix,
            )?,
            bias: weights
                .tensor(&format!("{prefix}.bias"), &[output_channels])?
                .to_vec(),
        })
    }

    fn forward(
        &self,
        input: &[f32],
        input_len: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        require_layout("conv1d input", input, self.input_channels, input_len)?;
        if input_len == 0 || self.kernel == 0 {
            return Err(VokraError::InvalidArgument(
                "bark/codec: conv1d requires non-empty input/kernel".to_owned(),
            ));
        }
        // Causal EnCodec: all fixed padding is on the left. Decoder Conv1D
        // layers have stride=dilation=1, so no extra right padding is needed.
        let padding_left = self.kernel - 1;
        let padded = reflect_pad1d(input, self.input_channels, input_len, padding_left, 0)?;
        let padded_len = input_len + padding_left;
        let output_len = padded_len - self.kernel + 1;
        let mut output = vec![0.0f32; self.output_channels * output_len];
        compute.conv1d_f32(
            &padded,
            self.input_channels,
            padded_len,
            &self.weight,
            self.output_channels,
            self.kernel,
            Some(&self.bias),
            1,
            0,
            &mut output,
        )?;
        Ok((output, output_len))
    }
}

#[derive(Debug)]
struct ConvTranspose1d {
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    /// Flattened PyTorch `[input, output, kernel]` as
    /// `[input, output*kernel]`, already in GEMM B layout.
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl ConvTranspose1d {
    fn bind_weight_norm(
        weights: &BarkMappedWeights,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self> {
        let magnitude = weights.tensor(&format!("{prefix}.weight_g"), &[input_channels, 1, 1])?;
        let direction = weights.tensor(
            &format!("{prefix}.weight_v"),
            &[input_channels, output_channels, kernel],
        )?;
        Ok(Self {
            input_channels,
            output_channels,
            kernel,
            stride,
            weight: reconstruct_weight_norm(
                magnitude,
                direction,
                input_channels,
                output_channels * kernel,
                prefix,
            )?,
            bias: weights
                .tensor(&format!("{prefix}.bias"), &[output_channels])?
                .to_vec(),
        })
    }

    fn forward(
        &self,
        input: &[f32],
        input_len: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        require_layout(
            "conv-transpose input",
            input,
            self.input_channels,
            input_len,
        )?;
        if input_len == 0 || self.stride == 0 || self.kernel < self.stride {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: invalid conv-transpose input_len={input_len}, kernel={}, stride={}",
                self.kernel, self.stride
            )));
        }
        let raw_len = (input_len - 1)
            .checked_mul(self.stride)
            .and_then(|value| value.checked_add(self.kernel))
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "bark/codec: conv-transpose output length overflow".to_owned(),
                )
            })?;
        let mut time_major = vec![0.0f32; input_len * self.input_channels];
        for time in 0..input_len {
            for channel in 0..self.input_channels {
                time_major[time * self.input_channels + channel] =
                    input[channel * input_len + time];
            }
        }
        let projected_width = self.output_channels * self.kernel;
        let mut projected = vec![0.0f32; input_len * projected_width];
        compute.gemm_f32(
            input_len,
            projected_width,
            self.input_channels,
            &time_major,
            &self.weight,
            None,
            &mut projected,
        )?;
        let mut raw = vec![0.0f32; self.output_channels * raw_len];
        for channel in 0..self.output_channels {
            raw[channel * raw_len..(channel + 1) * raw_len].fill(self.bias[channel]);
        }
        for time in 0..input_len {
            let destination = time * self.stride;
            for channel in 0..self.output_channels {
                let source = time * projected_width + channel * self.kernel;
                for tap in 0..self.kernel {
                    raw[channel * raw_len + destination + tap] += projected[source + tap];
                }
            }
        }

        // Causal + trim_right_ratio=1: remove the complete fixed padding from
        // the right and nothing from the left.
        let padding_right = self.kernel - self.stride;
        let output_len = raw_len.checked_sub(padding_right).ok_or_else(|| {
            VokraError::InvalidArgument(
                "bark/codec: conv-transpose right trim exceeds output".to_owned(),
            )
        })?;
        let mut output = vec![0.0f32; self.output_channels * output_len];
        for channel in 0..self.output_channels {
            output[channel * output_len..(channel + 1) * output_len]
                .copy_from_slice(&raw[channel * raw_len..channel * raw_len + output_len]);
        }
        Ok((output, output_len))
    }
}

#[derive(Debug)]
struct LstmLayer {
    weight_ih: Vec<f32>,
    weight_hh: Vec<f32>,
    bias_ih: Vec<f32>,
    bias_hh: Vec<f32>,
}

#[derive(Debug)]
struct Lstm {
    dimension: usize,
    layers: Vec<LstmLayer>,
}

impl Lstm {
    fn bind(
        weights: &BarkMappedWeights,
        prefix: &str,
        dimension: usize,
        layers: usize,
    ) -> Result<Self> {
        let gates = 4 * dimension;
        let mut bound = Vec::with_capacity(layers);
        for layer in 0..layers {
            bound.push(LstmLayer {
                weight_ih: weights
                    .tensor(&format!("{prefix}.weight_ih_l{layer}"), &[gates, dimension])?
                    .to_vec(),
                weight_hh: weights
                    .tensor(&format!("{prefix}.weight_hh_l{layer}"), &[gates, dimension])?
                    .to_vec(),
                bias_ih: weights
                    .tensor(&format!("{prefix}.bias_ih_l{layer}"), &[gates])?
                    .to_vec(),
                bias_hh: weights
                    .tensor(&format!("{prefix}.bias_hh_l{layer}"), &[gates])?
                    .to_vec(),
            });
        }
        Ok(Self {
            dimension,
            layers: bound,
        })
    }

    fn forward(&self, input: &[f32], time: usize, compute: &Compute) -> Result<Vec<f32>> {
        require_layout("LSTM input", input, self.dimension, time)?;
        let residual = input;
        let gates = 4 * self.dimension;
        let mut layer_input = input.to_vec();
        for layer in &self.layers {
            let mut output = vec![0.0f32; self.dimension * time];
            let mut hidden = vec![0.0f32; self.dimension];
            let mut cell = vec![0.0f32; self.dimension];
            let mut step_input = vec![0.0f32; self.dimension];
            let mut input_gates = vec![0.0f32; gates];
            let mut recurrent_gates = vec![0.0f32; gates];
            let mut candidates = vec![0.0f32; self.dimension];
            let mut tanh_cell = vec![0.0f32; self.dimension];
            for step in 0..time {
                for dimension in 0..self.dimension {
                    step_input[dimension] = layer_input[dimension * time + step];
                }
                compute.gemv_f32(
                    gates,
                    self.dimension,
                    &layer.weight_ih,
                    &step_input,
                    Some(&layer.bias_ih),
                    &mut input_gates,
                )?;
                compute.gemv_f32(
                    gates,
                    self.dimension,
                    &layer.weight_hh,
                    &hidden,
                    Some(&layer.bias_hh),
                    &mut recurrent_gates,
                )?;
                for dimension in 0..self.dimension {
                    candidates[dimension] = input_gates[2 * self.dimension + dimension]
                        + recurrent_gates[2 * self.dimension + dimension];
                }
                let candidate_source = candidates.clone();
                compute.tanh_f32(&candidate_source, &mut candidates)?;
                for dimension in 0..self.dimension {
                    let input_gate = sigmoid(input_gates[dimension] + recurrent_gates[dimension]);
                    let forget_gate = sigmoid(
                        input_gates[self.dimension + dimension]
                            + recurrent_gates[self.dimension + dimension],
                    );
                    cell[dimension] =
                        forget_gate * cell[dimension] + input_gate * candidates[dimension];
                }
                compute.tanh_f32(&cell, &mut tanh_cell)?;
                for dimension in 0..self.dimension {
                    let output_gate = sigmoid(
                        input_gates[3 * self.dimension + dimension]
                            + recurrent_gates[3 * self.dimension + dimension],
                    );
                    hidden[dimension] = output_gate * tanh_cell[dimension];
                    output[dimension * time + step] = hidden[dimension];
                }
            }
            layer_input = output;
        }
        add_assign(&mut layer_input, residual)?;
        Ok(layer_input)
    }
}

fn elu(compute: &Compute, input: &[f32]) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; input.len()];
    compute.elu_f32(input, &mut output)?;
    Ok(output)
}

fn reflect_pad1d(
    input: &[f32],
    channels: usize,
    length: usize,
    left: usize,
    right: usize,
) -> Result<Vec<f32>> {
    require_layout("reflect-pad input", input, channels, length)?;
    if length == 0 {
        return Err(VokraError::InvalidArgument(
            "bark/codec: reflect padding requires non-empty input".to_owned(),
        ));
    }
    let max_padding = left.max(right);
    let extra = if length <= max_padding {
        max_padding - length + 1
    } else {
        0
    };
    let base_len = length.checked_add(extra).ok_or_else(|| {
        VokraError::InvalidArgument("bark/codec: reflect padding base overflow".to_owned())
    })?;
    let padded_len = base_len
        .checked_add(left)
        .and_then(|value| value.checked_add(right))
        .ok_or_else(|| {
            VokraError::InvalidArgument("bark/codec: reflect padding output overflow".to_owned())
        })?;
    let output_len = padded_len - extra;
    let mut output = vec![0.0f32; channels * output_len];
    for channel in 0..channels {
        for output_index in 0..output_len {
            let logical = output_index as isize - left as isize;
            let source = if logical < 0 {
                (-logical) as usize
            } else if logical >= base_len as isize {
                (2 * base_len as isize - logical - 2) as usize
            } else {
                logical as usize
            };
            output[channel * output_len + output_index] = if source < length {
                input[channel * length + source]
            } else {
                0.0
            };
        }
    }
    Ok(output)
}

fn reconstruct_weight_norm(
    magnitude: &[f32],
    direction: &[f32],
    primary: usize,
    plane: usize,
    name: &str,
) -> Result<Vec<f32>> {
    if magnitude.len() != primary || direction.len() != primary * plane {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: weight-norm `{name}` has g/v lengths {}/{}, expected {primary}/{}",
            magnitude.len(),
            direction.len(),
            primary * plane
        )));
    }
    let mut output = vec![0.0f32; direction.len()];
    for row in 0..primary {
        let source = &direction[row * plane..(row + 1) * plane];
        let norm = source.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 || !magnitude[row].is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: weight-norm `{name}` row {row} has invalid norm/g {norm}/{}",
                magnitude[row]
            )));
        }
        let scale = magnitude[row] / norm;
        for (target, &value) in output[row * plane..(row + 1) * plane]
            .iter_mut()
            .zip(source)
        {
            *target = value * scale;
        }
    }
    reject_non_finite(name, &output).map_err(|error| VokraError::ModelLoad(error.to_string()))?;
    Ok(output)
}

fn require_layout(label: &str, values: &[f32], channels: usize, time: usize) -> Result<()> {
    let expected = channels.checked_mul(time).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{LABEL}: {label} shape overflows usize"))
    })?;
    if values.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {label} has {} values, expected {channels}x{time}={expected}",
            values.len()
        )));
    }
    Ok(())
}

fn add_assign(left: &mut [f32], right: &[f32]) -> Result<()> {
    if left.len() != right.len() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: residual length mismatch {} != {}",
            left.len(),
            right.len()
        )));
    }
    for (target, &value) in left.iter_mut().zip(right) {
        *target += value;
    }
    Ok(())
}

fn reject_non_finite(label: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {label} contains non-finite {value} at index {index}"
        )));
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
    fn codec_axes_match_the_public_causal_release() {
        assert_eq!(RATIOS.iter().product::<usize>(), FRAME_HOP);
        assert_eq!(LSTM_DIMENSION, NUM_FILTERS << RATIOS.len());
        assert_eq!(SAMPLE_RATE as usize / FRAME_HOP, 75);
    }

    #[test]
    fn causal_reflect_padding_is_left_only() {
        let padded = reflect_pad1d(&[1.0, 2.0, 3.0, 4.0], 1, 4, 3, 0).unwrap();
        assert_eq!(padded, vec![4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn weight_norm_is_per_primary_row() {
        let got = reconstruct_weight_norm(&[5.0, 13.0], &[3.0, 4.0, 5.0, 12.0], 2, 2, "x").unwrap();
        assert_eq!(got, vec![3.0, 4.0, 5.0, 12.0]);
    }
}
