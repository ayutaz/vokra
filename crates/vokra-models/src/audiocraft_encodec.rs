//! Native AudioCraft EnCodec token-to-waveform decoder.
//!
//! This module binds the EnCodec component already embedded in the public
//! Transformers-composite MusicGen Small/Melody GGUFs. It does **not** add a
//! standalone EnCodec converter, model-zoo entry, weight file, or publication
//! path: FR-OP-32's permanent standalone-weight exclusion remains unchanged.
//!
//! The fixed 32 kHz topology is transcribed from
//! facebook/musicgen-small/config.json and Transformers v4.45.2
//! modeling_encodec.py: four 2048-entry, 128-wide RVQ codebooks; a
//! non-causal weight-normalized SEANet decoder; ratios [8, 5, 4, 4]; and a
//! two-layer 1024-wide residual LSTM. Learned RVQ, convolution, transposed
//! convolution projection, and recurrent projections all dispatch through one
//! selected crate::compute::Compute backend. Reflect padding, layout
//! transposes, ELU/gate activations, scatter, and residual addition are
//! host-side tensor-layout glue, matching the established AudioSeal seam.
//! Unsupported backend coverage fails before any tensor is decoded; there is
//! no silent CPU fallback.
//!
//! The independent real-weight oracle is
//! tools/parity/audiocraft_encodec_dump_reference.py. It imports the pinned
//! official Transformers forward and is intentionally run only on VAST; no
//! model payload is loaded on the maintainer Mac.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};
use vokra_ops::{CodebookTable, EncodecRvqAttrs};

use crate::compute::{Compute, HotOp};

/// Sample rate of the EnCodec component embedded in MusicGen.
pub const SAMPLE_RATE: u32 = 32_000;
/// Number of PCM samples emitted per codec frame.
pub const FRAME_HOP: usize = 640;
/// Number of residual codebooks consumed per frame.
pub const NUM_CODEBOOKS: usize = 4;
/// Entries in each residual codebook.
pub const CODEBOOK_SIZE: usize = 2_048;
/// EnCodec latent width.
pub const DIMENSION: usize = 128;

const PREFIX: &str = "audio_encoder";
const NUM_FILTERS: usize = 64;
const LSTM_DIMENSION: usize = 1_024;
const LSTM_LAYERS: usize = 2;
const RATIOS: [usize; 4] = [8, 5, 4, 4];

/// Complete learned-op set for the public MusicGen EnCodec decode path.
pub const AUDIOCRAFT_ENCODEC_HOT_OPS: &[HotOp] =
    &[HotOp::EncodecRvq, HotOp::Conv1d, HotOp::Gemm, HotOp::Gemv];

/// The native 32 kHz EnCodec decoder embedded in a MusicGen composite GGUF.
#[derive(Debug)]
pub struct AudioCraftEncodecDecoder {
    backend: BackendKind,
    codebooks: Vec<CodebookTable>,
    decoder: SeanetDecoder,
}

impl AudioCraftEncodecDecoder {
    /// Binds the authenticated Transformers-composite tensor names.
    ///
    /// The caller authenticates the complete MusicGen artifact manifest before
    /// entering this component binder. Every tensor used here is shape-checked
    /// again before its payload is decoded.
    pub(crate) fn bind_transformers_composite(
        file: &GgufFile,
        backend: BackendKind,
    ) -> Result<Self> {
        // Fail backend coverage before reading any learned tensor payload.
        let _ = Compute::for_backend(backend, AUDIOCRAFT_ENCODEC_HOT_OPS)?;

        let mut codebooks = Vec::with_capacity(NUM_CODEBOOKS);
        for index in 0..NUM_CODEBOOKS {
            let name = format!("{PREFIX}.quantizer.layers.{index}.codebook.embed");
            let values = tensor(file, &name, &[CODEBOOK_SIZE, DIMENSION])?;
            codebooks.push(CodebookTable::new(CODEBOOK_SIZE, DIMENSION, values)?);
        }

        Ok(Self {
            backend,
            codebooks,
            decoder: SeanetDecoder::load_transformers(file)?,
        })
    }

    /// Selected backend for every learned operation in the decode path.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Decodes frame-major [frames, 4] EnCodec indices to mono 32 kHz PCM.
    pub fn decode_frame_major(&self, codes: &[u32], frames: usize) -> Result<Vec<f32>> {
        if frames == 0 {
            return Err(VokraError::InvalidArgument(
                "audiocraft encodec: frames must be > 0".to_owned(),
            ));
        }
        let expected_codes = frames.checked_mul(NUM_CODEBOOKS).ok_or_else(|| {
            VokraError::InvalidArgument(
                "audiocraft encodec: frames * num_codebooks overflows usize".to_owned(),
            )
        })?;
        if codes.len() != expected_codes {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft encodec: codes.len() {} != frames {frames} * \
                 num_codebooks {NUM_CODEBOOKS} = {expected_codes}",
                codes.len()
            )));
        }

        let compute = Compute::for_backend(self.backend, AUDIOCRAFT_ENCODEC_HOT_OPS)?;
        let latent_time_major = compute.encodec_rvq_f32(
            codes,
            frames,
            &self.codebooks,
            &EncodecRvqAttrs {
                n_codebooks: NUM_CODEBOOKS,
                codebook_size: CODEBOOK_SIZE,
                d_model: DIMENSION,
            },
        )?;
        let mut latent = vec![0.0f32; latent_time_major.len()];
        for frame in 0..frames {
            for channel in 0..DIMENSION {
                latent[channel * frames + frame] = latent_time_major[frame * DIMENSION + channel];
            }
        }

        let (pcm, samples) = self.decoder.forward(&latent, frames, &compute)?;
        let expected_samples = frames.checked_mul(FRAME_HOP).ok_or_else(|| {
            VokraError::InvalidArgument(
                "audiocraft encodec: frames * frame_hop overflows usize".to_owned(),
            )
        })?;
        if samples != expected_samples || pcm.len() != expected_samples {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft encodec: decoder emitted {} values / {samples} samples, \
                 expected {expected_samples}",
                pcm.len()
            )));
        }
        reject_non_finite("decoded PCM", &pcm)?;
        Ok(pcm)
    }
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
    fn load_transformers(file: &GgufFile) -> Result<Self> {
        let decoder = format!("{PREFIX}.decoder.layers");
        let initial = Conv1d::load_weight_norm(
            file,
            &format!("{decoder}.0.conv"),
            DIMENSION,
            LSTM_DIMENSION,
            7,
        )?;
        let lstm = Lstm::load(
            file,
            &format!("{decoder}.1.lstm"),
            LSTM_DIMENSION,
            LSTM_LAYERS,
        )?;

        let mut upsample = Vec::with_capacity(RATIOS.len());
        let mut residuals = Vec::with_capacity(RATIOS.len());
        for (transpose, residual, channels, next, ratio) in [
            (3, 4, 1_024, 512, 8),
            (6, 7, 512, 256, 5),
            (9, 10, 256, 128, 4),
            (12, 13, 128, 64, 4),
        ] {
            upsample.push(ConvTranspose1d::load_weight_norm(
                file,
                &format!("{decoder}.{transpose}.conv"),
                channels,
                next,
                ratio * 2,
                ratio,
            )?);
            residuals.push(ResidualBlock::load(
                file,
                &format!("{decoder}.{residual}"),
                next,
            )?);
        }
        Ok(Self {
            initial,
            lstm,
            upsample,
            residuals,
            final_conv: Conv1d::load_weight_norm(
                file,
                &format!("{decoder}.15.conv"),
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
            elu_inplace(&mut hidden);
            (hidden, time) = self.upsample[stage].forward(&hidden, time, compute)?;
            hidden = self.residuals[stage].forward(&hidden, time, compute)?;
        }
        elu_inplace(&mut hidden);
        self.final_conv.forward(&hidden, time, compute)
    }
}

#[derive(Debug)]
struct ResidualBlock {
    first: Conv1d,
    second: Conv1d,
}

impl ResidualBlock {
    fn load(file: &GgufFile, prefix: &str, channels: usize) -> Result<Self> {
        let hidden = channels / 2;
        Ok(Self {
            first: Conv1d::load_weight_norm(
                file,
                &format!("{prefix}.block.1.conv"),
                channels,
                hidden,
                3,
            )?,
            second: Conv1d::load_weight_norm(
                file,
                &format!("{prefix}.block.3.conv"),
                hidden,
                channels,
                1,
            )?,
        })
    }

    fn forward(&self, input: &[f32], time: usize, compute: &Compute) -> Result<Vec<f32>> {
        require_layout("residual input", input, self.first.input_channels, time)?;
        let residual = input;
        let mut hidden = input.to_vec();
        elu_inplace(&mut hidden);
        let (mut hidden, first_time) = self.first.forward(&hidden, time, compute)?;
        if first_time != time {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft encodec: residual k=3 convolution changed time \
                 {time} -> {first_time}"
            )));
        }
        elu_inplace(&mut hidden);
        let (mut hidden, second_time) = self.second.forward(&hidden, time, compute)?;
        if second_time != time || hidden.len() != residual.len() {
            return Err(VokraError::InvalidArgument(format!(
                "audiocraft encodec: residual branch shape mismatch (time {second_time}, \
                 values {}) vs input (time {time}, values {})",
                hidden.len(),
                residual.len()
            )));
        }
        for (value, &skip) in hidden.iter_mut().zip(residual) {
            *value += skip;
        }
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
    fn load_weight_norm(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
    ) -> Result<Self> {
        let g = tensor(
            file,
            &format!("{prefix}.weight_g"),
            &[output_channels, 1, 1],
        )?;
        let v = tensor(
            file,
            &format!("{prefix}.weight_v"),
            &[output_channels, input_channels, kernel],
        )?;
        Ok(Self {
            input_channels,
            output_channels,
            kernel,
            weight: reconstruct_weight_norm(
                &g,
                &v,
                output_channels,
                input_channels * kernel,
                prefix,
            )?,
            bias: tensor(file, &format!("{prefix}.bias"), &[output_channels])?,
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
                "audiocraft encodec: conv1d requires non-empty input/kernel".to_owned(),
            ));
        }
        // Every fixed decoder Conv1D has stride=1 and dilation=1. Transformers
        // therefore reflect-pads floor/ceil((kernel-1)/2) and preserves time.
        let padding_total = self.kernel - 1;
        let padding_right = padding_total / 2;
        let padding_left = padding_total - padding_right;
        let padded = reflect_pad1d(
            input,
            self.input_channels,
            input_len,
            padding_left,
            padding_right,
        )?;
        let padded_len = input_len + padding_total;
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
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl ConvTranspose1d {
    #[allow(clippy::too_many_arguments)]
    fn load_weight_norm(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self> {
        let g = tensor(file, &format!("{prefix}.weight_g"), &[input_channels, 1, 1])?;
        let v = tensor(
            file,
            &format!("{prefix}.weight_v"),
            &[input_channels, output_channels, kernel],
        )?;
        Ok(Self {
            input_channels,
            output_channels,
            kernel,
            stride,
            weight: reconstruct_weight_norm(
                &g,
                &v,
                input_channels,
                output_channels * kernel,
                prefix,
            )?,
            bias: tensor(file, &format!("{prefix}.bias"), &[output_channels])?,
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
                "audiocraft encodec: invalid conv-transpose input_len={input_len}, \
                 kernel={}, stride={}",
                self.kernel, self.stride
            )));
        }
        let raw_len = (input_len - 1)
            .checked_mul(self.stride)
            .and_then(|value| value.checked_add(self.kernel))
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "audiocraft encodec: conv-transpose output length overflow".to_owned(),
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

        // Non-causal EnCodec asymmetric trim: right=floor((k-s)/2),
        // left=(k-s)-right. For k=2s this emits exactly input_len*stride.
        let padding_total = self.kernel - self.stride;
        let padding_right = padding_total / 2;
        let padding_left = padding_total - padding_right;
        let output_len = raw_len
            .checked_sub(padding_left + padding_right)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "audiocraft encodec: conv-transpose trim exceeds output".to_owned(),
                )
            })?;
        let mut output = vec![0.0f32; self.output_channels * output_len];
        for channel in 0..self.output_channels {
            output[channel * output_len..(channel + 1) * output_len].copy_from_slice(
                &raw[channel * raw_len + padding_left
                    ..channel * raw_len + padding_left + output_len],
            );
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
    fn load(file: &GgufFile, prefix: &str, dimension: usize, layers: usize) -> Result<Self> {
        let gates = 4 * dimension;
        let mut bound = Vec::with_capacity(layers);
        for layer in 0..layers {
            bound.push(LstmLayer {
                weight_ih: tensor(
                    file,
                    &format!("{prefix}.weight_ih_l{layer}"),
                    &[gates, dimension],
                )?,
                weight_hh: tensor(
                    file,
                    &format!("{prefix}.weight_hh_l{layer}"),
                    &[gates, dimension],
                )?,
                bias_ih: tensor(file, &format!("{prefix}.bias_ih_l{layer}"), &[gates])?,
                bias_hh: tensor(file, &format!("{prefix}.bias_hh_l{layer}"), &[gates])?,
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
                    let input_gate = sigmoid(input_gates[dimension] + recurrent_gates[dimension]);
                    let forget_gate = sigmoid(
                        input_gates[self.dimension + dimension]
                            + recurrent_gates[self.dimension + dimension],
                    );
                    let candidate = (input_gates[2 * self.dimension + dimension]
                        + recurrent_gates[2 * self.dimension + dimension])
                        .tanh();
                    let output_gate = sigmoid(
                        input_gates[3 * self.dimension + dimension]
                            + recurrent_gates[3 * self.dimension + dimension],
                    );
                    cell[dimension] = forget_gate * cell[dimension] + input_gate * candidate;
                    hidden[dimension] = output_gate * cell[dimension].tanh();
                    output[dimension * time + step] = hidden[dimension];
                }
            }
            layer_input = output;
        }
        for (value, &skip) in layer_input.iter_mut().zip(residual) {
            *value += skip;
        }
        Ok(layer_input)
    }
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
            "audiocraft encodec: reflect padding requires non-empty input".to_owned(),
        ));
    }
    let max_padding = left.max(right);
    let extra = if length <= max_padding {
        max_padding - length + 1
    } else {
        0
    };
    let base_len = length.checked_add(extra).ok_or_else(|| {
        VokraError::InvalidArgument(
            "audiocraft encodec: reflect padding base length overflow".to_owned(),
        )
    })?;
    let padded_len = base_len
        .checked_add(left)
        .and_then(|value| value.checked_add(right))
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "audiocraft encodec: reflect padding output length overflow".to_owned(),
            )
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
    g: &[f32],
    v: &[f32],
    primary: usize,
    plane: usize,
    name: &str,
) -> Result<Vec<f32>> {
    if g.len() != primary || v.len() != primary * plane {
        return Err(VokraError::ModelLoad(format!(
            "audiocraft encodec: weight-norm '{name}' has g/v lengths {}/{}, \
             expected {primary}/{}",
            g.len(),
            v.len(),
            primary * plane
        )));
    }
    let mut output = vec![0.0f32; v.len()];
    for row in 0..primary {
        let source = &v[row * plane..(row + 1) * plane];
        let norm = source.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 || !g[row].is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "audiocraft encodec: weight-norm '{name}' row {row} has \
                 invalid norm/g {norm}/{}",
                g[row]
            )));
        }
        let scale = g[row] / norm;
        for (destination, &value) in output[row * plane..(row + 1) * plane]
            .iter_mut()
            .zip(source)
        {
            *destination = value * scale;
        }
    }
    reject_non_finite(name, &output).map_err(|error| VokraError::ModelLoad(error.to_string()))?;
    Ok(output)
}

fn tensor(file: &GgufFile, name: &str, dimensions: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("audiocraft encodec: missing '{name}'")))?;
    let expected = dimensions
        .iter()
        .map(|&dimension| dimension as u64)
        .collect::<Vec<_>>();
    if info.dimensions != expected {
        return Err(VokraError::ModelLoad(format!(
            "audiocraft encodec: tensor '{name}' shape {:?}, expected {expected:?}",
            info.dimensions
        )));
    }
    let values = file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!(
            "audiocraft encodec: tensor '{name}' decode failed: {error}"
        ))
    })?;
    reject_non_finite(name, &values).map_err(|error| VokraError::ModelLoad(error.to_string()))?;
    Ok(values)
}

fn require_layout(label: &str, values: &[f32], channels: usize, time: usize) -> Result<()> {
    let expected = channels.checked_mul(time).ok_or_else(|| {
        VokraError::InvalidArgument(format!("audiocraft encodec: {label} shape overflows usize"))
    })?;
    if values.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "audiocraft encodec: {label} has {} values, expected \
             {channels}x{time}={expected}",
            values.len()
        )));
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
            "audiocraft encodec: {label} contains non-finite {value} at index {index}"
        )));
    }
    Ok(())
}

fn elu_inplace(values: &mut [f32]) {
    for value in values {
        if *value < 0.0 {
            *value = value.exp_m1();
        }
    }
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
    fn fixed_contract_matches_the_public_musicgen_codec() {
        assert_eq!(RATIOS.iter().product::<usize>(), FRAME_HOP);
        assert_eq!(LSTM_DIMENSION, NUM_FILTERS << RATIOS.len());
        assert_eq!(AUDIOCRAFT_ENCODEC_HOT_OPS.len(), 4);
    }

    #[test]
    fn reflect_padding_matches_torch_for_normal_and_short_inputs() {
        let normal = reflect_pad1d(&[1.0, 2.0, 3.0, 4.0], 1, 4, 2, 2).unwrap();
        assert_eq!(normal, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);

        // Transformers' short-input helper first appends zeros until reflect
        // padding is legal, then removes that temporary tail from the result.
        let short = reflect_pad1d(&[7.0], 1, 1, 3, 3).unwrap();
        assert_eq!(short, vec![0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn weight_norm_is_per_primary_row() {
        let got = reconstruct_weight_norm(&[5.0, 13.0], &[3.0, 4.0, 5.0, 12.0], 2, 2, "x").unwrap();
        assert_eq!(got, vec![3.0, 4.0, 5.0, 12.0]);
    }
}
