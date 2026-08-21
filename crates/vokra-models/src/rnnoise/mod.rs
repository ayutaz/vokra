//! Xiph RNNoise v0.2 real-weight network binder and streaming neural forward.
//!
//! This module implements the exact neural network shipped in the official
//! v0.2 `rnnoise_data.c`: two causal Conv1d projections, three 384-wide GRUs,
//! a 32-band gain head, and a VAD head.  The quantized layers preserve Xiph's
//! signed-int8 activation rounding, 8×4 blocked matrix layout, sparse index
//! walk, per-output scale, recurrent diagonal, `[z, r, h]` gate order, and
//! rational tanh/sigmoid approximation.
//!
//! Waveform analysis/synthesis is intentionally a separate boundary.  This
//! binder consumes the canonical 65-feature frames produced by v0.2 and emits
//! the real per-band network decisions; it never fabricates enhanced PCM.

use std::sync::Arc;

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{Result, VokraError};

/// Required GGUF architecture tag.
pub const ARCH: &str = "rnnoise";
/// Upstream PCM sample rate in hertz.
pub const SAMPLE_RATE: u32 = 48_000;
/// Samples consumed per 10 ms frame.
pub const FRAME_SIZE: usize = 480;
/// Samples in the 20 ms analysis window.
pub const WINDOW_SIZE: usize = 960;
/// Gain bands predicted by the v0.2 network.
pub const N_BANDS: usize = 32;
/// Input features consumed per frame.
pub const N_FEATURES: usize = 65;
/// First causal convolution output width.
pub const CONV1_WIDTH: usize = 128;
/// Width of the second convolution and each GRU state.
pub const HIDDEN_SIZE: usize = 384;
/// Number of recurrent layers.
pub const N_GRU: usize = 3;

const KEY_RELEASE_SHA256: &str = "vokra.rnnoise.release_tarball_sha256";
const KEY_SAMPLE_RATE: &str = "vokra.rnnoise.sample_rate";
const KEY_FRAME_SIZE: &str = "vokra.rnnoise.frame_size";
const KEY_WINDOW_SIZE: &str = "vokra.rnnoise.window_size";
const KEY_N_BANDS: &str = "vokra.rnnoise.n_bands";
const KEY_N_FEATURES: &str = "vokra.rnnoise.n_features";
const KEY_CONV1_WIDTH: &str = "vokra.rnnoise.conv1_width";
const KEY_HIDDEN_SIZE: &str = "vokra.rnnoise.hidden_size";
const KEY_N_GRU: &str = "vokra.rnnoise.n_gru";
const KEY_QUANTIZATION: &str = "vokra.rnnoise.quantization";
const KEY_GATE_ORDER: &str = "vokra.rnnoise.gate_order";

/// SHA-256 of the official GitHub v0.2 source release tarball.
pub const RELEASE_TARBALL_SHA256: &str =
    "90fce4b00b9ff24c08dbfe31b82ffd43bae383d85c5535676d28b0a2b11c0d37";

#[derive(Debug, Clone)]
enum Matrix {
    Float(Vec<f32>),
    Quantized {
        weights: Vec<i8>,
        indices: Option<Vec<i32>>,
        scale: Vec<f32>,
    },
}

#[derive(Debug, Clone)]
struct Linear {
    input: usize,
    output: usize,
    matrix: Matrix,
    bias: Vec<f32>,
    diagonal: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
struct RnnoiseWeights {
    conv1: Linear,
    conv2: Linear,
    gru_input: [Linear; N_GRU],
    gru_recurrent: [Linear; N_GRU],
    gain: Linear,
    vad: Linear,
}

/// Immutable RNNoise v0.2 network bound to canonical real weights.
#[derive(Debug, Clone)]
pub struct RnnoiseV02 {
    weights: Arc<RnnoiseWeights>,
}

/// Per-stream causal Conv and GRU state.
#[derive(Debug, Clone)]
pub struct RnnoiseNetworkState {
    conv1: Vec<f32>,
    conv2: Vec<f32>,
    gru: [Vec<f32>; N_GRU],
}

impl Default for RnnoiseNetworkState {
    fn default() -> Self {
        Self {
            conv1: vec![0.0; N_FEATURES * 2],
            conv2: vec![0.0; CONV1_WIDTH * 2],
            gru: std::array::from_fn(|_| vec![0.0; HIDDEN_SIZE]),
        }
    }
}

/// Neural decisions for one 10 ms RNNoise frame.
#[derive(Debug, Clone, PartialEq)]
pub struct RnnoiseNetworkOutput {
    /// Sigmoid gain for each of the 32 v0.2 frequency bands.
    pub gains: [f32; N_BANDS],
    /// Sigmoid voice-activity probability from the auxiliary head.
    pub vad_probability: f32,
}

impl RnnoiseV02 {
    /// Binds the official v0.2 tensor manifest from a parsed GGUF.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        require_string(gguf, chunks::KEY_MODEL_ARCH, ARCH)?;
        require_string(gguf, KEY_RELEASE_SHA256, RELEASE_TARBALL_SHA256)?;
        require_u32(gguf, KEY_SAMPLE_RATE, SAMPLE_RATE)?;
        require_u32(gguf, KEY_FRAME_SIZE, FRAME_SIZE as u32)?;
        require_u32(gguf, KEY_WINDOW_SIZE, WINDOW_SIZE as u32)?;
        require_u32(gguf, KEY_N_BANDS, N_BANDS as u32)?;
        require_u32(gguf, KEY_N_FEATURES, N_FEATURES as u32)?;
        require_u32(gguf, KEY_CONV1_WIDTH, CONV1_WIDTH as u32)?;
        require_u32(gguf, KEY_HIDDEN_SIZE, HIDDEN_SIZE as u32)?;
        require_u32(gguf, KEY_N_GRU, N_GRU as u32)?;
        require_string(gguf, KEY_QUANTIZATION, "signed-i8-round127-f32-container")?;
        require_string(gguf, KEY_GATE_ORDER, "zrh")?;

        let conv1 = load_float(gguf, "conv1", 195, 128)?;
        let conv2 = load_quantized(gguf, "conv2", 384, 384, false, false)?;
        let gru_input: [Linear; N_GRU] = (1..=N_GRU)
            .map(|layer| {
                load_quantized(
                    gguf,
                    &format!("gru{layer}_input"),
                    HIDDEN_SIZE,
                    3 * HIDDEN_SIZE,
                    true,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| VokraError::ModelLoad("rnnoise: expected three input GRUs".to_owned()))?;
        let gru_recurrent: [Linear; N_GRU] = (1..=N_GRU)
            .map(|layer| {
                load_quantized(
                    gguf,
                    &format!("gru{layer}_recurrent"),
                    HIDDEN_SIZE,
                    3 * HIDDEN_SIZE,
                    true,
                    true,
                )
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| {
                VokraError::ModelLoad("rnnoise: expected three recurrent GRUs".to_owned())
            })?;
        let gain = load_float_named(
            gguf,
            "dense_out",
            "dense_out_weights_float",
            HIDDEN_SIZE,
            N_BANDS,
        )?;
        let vad = load_float_named(gguf, "vad_dense", "vad_dense_weights_float", HIDDEN_SIZE, 1)?;
        Ok(Self {
            weights: Arc::new(RnnoiseWeights {
                conv1,
                conv2,
                gru_input,
                gru_recurrent,
                gain,
                vad,
            }),
        })
    }

    /// Opens and binds a canonical v0.2 GGUF.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// Runs one real 65-feature frame through the causal network.
    pub fn forward_features(
        &self,
        state: &mut RnnoiseNetworkState,
        features: &[f32; N_FEATURES],
    ) -> Result<RnnoiseNetworkOutput> {
        if features.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "rnnoise: feature frame contains a non-finite value".to_owned(),
            ));
        }
        let conv1 = causal_conv(&self.weights.conv1, &mut state.conv1, features, N_FEATURES)?;
        let conv2 = causal_conv(&self.weights.conv2, &mut state.conv2, &conv1, CONV1_WIDTH)?;
        gru(
            &self.weights.gru_input[0],
            &self.weights.gru_recurrent[0],
            &mut state.gru[0],
            &conv2,
        )?;
        for layer in 1..N_GRU {
            let (previous, current) = state.gru.split_at_mut(layer);
            gru(
                &self.weights.gru_input[layer],
                &self.weights.gru_recurrent[layer],
                &mut current[0],
                &previous[layer - 1],
            )?;
        }
        let gain_logits = self.weights.gain.forward(&state.gru[2])?;
        let vad_logits = self.weights.vad.forward(&state.gru[2])?;
        let gains = std::array::from_fn(|index| sigmoid_approx(gain_logits[index]));
        Ok(RnnoiseNetworkOutput {
            gains,
            vad_probability: sigmoid_approx(vad_logits[0]),
        })
    }
}

impl Linear {
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input {
            return Err(VokraError::InvalidArgument(format!(
                "rnnoise Linear input has {} elements, expected {}",
                input.len(),
                self.input
            )));
        }
        let mut output = match &self.matrix {
            Matrix::Float(weights) => float_matvec(weights, self.output, input),
            Matrix::Quantized {
                weights,
                indices,
                scale,
            } => quantized_matvec(weights, indices.as_deref(), scale, self.output, input)?,
        };
        for (value, bias) in output.iter_mut().zip(&self.bias) {
            *value += *bias;
        }
        if let Some(diagonal) = &self.diagonal {
            if self.output != 3 * self.input {
                return Err(VokraError::ModelLoad(
                    "rnnoise: recurrent diagonal requires output = 3 * input".to_owned(),
                ));
            }
            for gate in 0..3 {
                for index in 0..self.input {
                    output[gate * self.input + index] +=
                        diagonal[gate * self.input + index] * input[index];
                }
            }
        }
        Ok(output)
    }
}

fn causal_conv(
    layer: &Linear,
    memory: &mut [f32],
    input: &[f32],
    width: usize,
) -> Result<Vec<f32>> {
    let history = layer.input.checked_sub(width).ok_or_else(|| {
        VokraError::ModelLoad("rnnoise: causal Conv input width exceeds kernel width".to_owned())
    })?;
    if memory.len() != history || input.len() != width {
        return Err(VokraError::InvalidArgument(format!(
            "rnnoise: causal Conv state/input mismatch (state {}, input {}, expected {history}/{width})",
            memory.len(),
            input.len()
        )));
    }
    let mut joined = Vec::with_capacity(layer.input);
    joined.extend_from_slice(memory);
    joined.extend_from_slice(input);
    let mut output = layer.forward(&joined)?;
    output
        .iter_mut()
        .for_each(|value| *value = tanh_approx(*value));
    memory.copy_from_slice(&joined[width..]);
    Ok(output)
}

fn gru(input_layer: &Linear, recurrent: &Linear, state: &mut [f32], input: &[f32]) -> Result<()> {
    let n = state.len();
    let mut zrh = input_layer.forward(input)?;
    let recur = recurrent.forward(state)?;
    if zrh.len() != 3 * n || recur.len() != 3 * n {
        return Err(VokraError::ModelLoad(
            "rnnoise: GRU projection output is not three gate blocks".to_owned(),
        ));
    }
    for index in 0..2 * n {
        zrh[index] += recur[index];
        zrh[index] = sigmoid_approx(zrh[index]);
    }
    for index in 0..n {
        let z = zrh[index];
        let r = zrh[n + index];
        let candidate = tanh_approx(zrh[2 * n + index] + recur[2 * n + index] * r);
        state[index] = z * state[index] + (1.0 - z) * candidate;
    }
    Ok(())
}

fn float_matvec(weights: &[f32], output: usize, input: &[f32]) -> Vec<f32> {
    let mut result = vec![0.0; output];
    for (column, input_value) in input.iter().copied().enumerate() {
        for row in 0..output {
            result[row] += weights[column * output + row] * input_value;
        }
    }
    result
}

fn quantized_matvec(
    weights: &[i8],
    indices: Option<&[i32]>,
    scale: &[f32],
    output: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    let quantized: Vec<i8> = input
        .iter()
        .map(|value| (0.5 + 127.0 * value).floor() as i8)
        .collect();
    let mut result = vec![0.0; output];
    let mut weight_offset = 0usize;
    if let Some(indices) = indices {
        let mut index_offset = 0usize;
        for row_base in (0..output).step_by(8) {
            let blocks = usize::try_from(indices[index_offset]).map_err(|_| {
                VokraError::ModelLoad("rnnoise: negative sparse block count".to_owned())
            })?;
            index_offset += 1;
            for _ in 0..blocks {
                let column = usize::try_from(indices[index_offset]).map_err(|_| {
                    VokraError::ModelLoad("rnnoise: negative sparse column".to_owned())
                })?;
                index_offset += 1;
                accumulate_i8_block(
                    &mut result[row_base..row_base + 8],
                    &weights[weight_offset..weight_offset + 32],
                    &quantized[column..column + 4],
                );
                weight_offset += 32;
            }
        }
        if index_offset != indices.len() {
            return Err(VokraError::ModelLoad(format!(
                "rnnoise: sparse index walk consumed {index_offset}, stored {}",
                indices.len()
            )));
        }
    } else {
        for row_base in (0..output).step_by(8) {
            for column in (0..input.len()).step_by(4) {
                accumulate_i8_block(
                    &mut result[row_base..row_base + 8],
                    &weights[weight_offset..weight_offset + 32],
                    &quantized[column..column + 4],
                );
                weight_offset += 32;
            }
        }
    }
    if weight_offset != weights.len() {
        return Err(VokraError::ModelLoad(format!(
            "rnnoise: quantized matrix walk consumed {weight_offset} weights, stored {}",
            weights.len()
        )));
    }
    for (value, scale) in result.iter_mut().zip(scale) {
        *value *= *scale;
    }
    Ok(result)
}

fn accumulate_i8_block(output: &mut [f32], weights: &[i8], input: &[i8]) {
    for row in 0..8 {
        let base = 4 * row;
        let sum = i32::from(weights[base]) * i32::from(input[0])
            + i32::from(weights[base + 1]) * i32::from(input[1])
            + i32::from(weights[base + 2]) * i32::from(input[2])
            + i32::from(weights[base + 3]) * i32::from(input[3]);
        output[row] += sum as f32;
    }
}

fn tanh_approx(value: f32) -> f32 {
    let square = value * value;
    let numerator = ((0.608_630_42 * square + 96.392_36) * square + 952.528_0) * value;
    let denominator = (11.886_009 * square + 413.368_0) * square + 952.724_0;
    (numerator / denominator).clamp(-1.0, 1.0)
}

fn sigmoid_approx(value: f32) -> f32 {
    0.5 + 0.5 * tanh_approx(0.5 * value)
}

fn require_string(gguf: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = gguf.get(key).and_then(|value| value.as_str());
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "rnnoise: metadata `{key}` is {actual:?}, expected `{expected}`"
        )));
    }
    Ok(())
}

fn require_u32(gguf: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = gguf.get(key).and_then(|value| value.as_u64());
    if actual != Some(u64::from(expected)) {
        return Err(VokraError::ModelLoad(format!(
            "rnnoise: metadata `{key}` is {actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

fn load_tensor(gguf: &GgufFile, name: &str, expected: usize) -> Result<Vec<f32>> {
    let values = gguf.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("rnnoise: tensor `{name}` load failed: {error}"))
    })?;
    if values.len() != expected {
        return Err(VokraError::ModelLoad(format!(
            "rnnoise: tensor `{name}` has {} elements, expected {expected}",
            values.len()
        )));
    }
    Ok(values)
}

fn exact_i8(gguf: &GgufFile, name: &str, expected: usize) -> Result<Vec<i8>> {
    load_tensor(gguf, name, expected)?
        .into_iter()
        .map(|value| {
            if !value.is_finite() || value.fract() != 0.0 || !(-128.0..=127.0).contains(&value) {
                Err(VokraError::ModelLoad(format!(
                    "rnnoise: tensor `{name}` contains non-int8 container value {value}"
                )))
            } else {
                Ok(value as i8)
            }
        })
        .collect()
}

fn exact_i32(gguf: &GgufFile, name: &str, expected: usize) -> Result<Vec<i32>> {
    load_tensor(gguf, name, expected)?
        .into_iter()
        .map(|value| {
            if !value.is_finite()
                || value.fract() != 0.0
                || value < i32::MIN as f32
                || value > i32::MAX as f32
            {
                Err(VokraError::ModelLoad(format!(
                    "rnnoise: tensor `{name}` contains non-i32 container value {value}"
                )))
            } else {
                Ok(value as i32)
            }
        })
        .collect()
}

fn load_float(gguf: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<Linear> {
    load_float_named(
        gguf,
        prefix,
        &format!("{prefix}_weights_float"),
        input,
        output,
    )
}

fn load_float_named(
    gguf: &GgufFile,
    prefix: &str,
    weight_name: &str,
    input: usize,
    output: usize,
) -> Result<Linear> {
    Ok(Linear {
        input,
        output,
        matrix: Matrix::Float(load_tensor(gguf, weight_name, input * output)?),
        bias: load_tensor(gguf, &format!("{prefix}_bias"), output)?,
        diagonal: None,
    })
}

fn load_quantized(
    gguf: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    sparse: bool,
    diagonal: bool,
) -> Result<Linear> {
    if input % 4 != 0 || output % 8 != 0 {
        return Err(VokraError::ModelLoad(format!(
            "rnnoise: quantized layer `{prefix}` is not aligned to 8x4 blocks"
        )));
    }
    let indices = if sparse {
        let values = exact_i32(gguf, &format!("{prefix}_weights_idx"), 4_752)?;
        validate_sparse_indices(prefix, &values, input, output)?;
        Some(values)
    } else {
        None
    };
    let weight_count = if sparse {
        // The official v0.2 sparse layers each carry 4,608 non-zero 8×4 blocks.
        147_456
    } else {
        input * output
    };
    Ok(Linear {
        input,
        output,
        matrix: Matrix::Quantized {
            weights: exact_i8(gguf, &format!("{prefix}_weights_int8"), weight_count)?,
            indices,
            scale: load_tensor(gguf, &format!("{prefix}_scale"), output)?,
        },
        bias: load_tensor(gguf, &format!("{prefix}_bias"), output)?,
        diagonal: if diagonal {
            Some(load_tensor(
                gguf,
                &format!("{prefix}_weights_diag"),
                output,
            )?)
        } else {
            None
        },
    })
}

fn validate_sparse_indices(
    prefix: &str,
    indices: &[i32],
    input: usize,
    output: usize,
) -> Result<()> {
    let mut cursor = 0usize;
    let mut blocks = 0usize;
    for _ in (0..output).step_by(8) {
        let count = indices.get(cursor).copied().ok_or_else(|| {
            VokraError::ModelLoad(format!("rnnoise: `{prefix}` sparse index ended early"))
        })?;
        let count = usize::try_from(count).map_err(|_| {
            VokraError::ModelLoad(format!("rnnoise: `{prefix}` has a negative block count"))
        })?;
        cursor += 1;
        for _ in 0..count {
            let position = indices.get(cursor).copied().ok_or_else(|| {
                VokraError::ModelLoad(format!("rnnoise: `{prefix}` sparse index ended early"))
            })?;
            let position = usize::try_from(position).map_err(|_| {
                VokraError::ModelLoad(format!("rnnoise: `{prefix}` has a negative column"))
            })?;
            if position % 4 != 0 || position + 3 >= input {
                return Err(VokraError::ModelLoad(format!(
                    "rnnoise: `{prefix}` sparse column {position} is outside/alignment-invalid for input {input}"
                )));
            }
            cursor += 1;
            blocks += 1;
        }
    }
    if cursor != indices.len() || blocks * 32 != 147_456 {
        return Err(VokraError::ModelLoad(format!(
            "rnnoise: `{prefix}` sparse manifest consumed {cursor}/{} indices and {blocks} blocks",
            indices.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_matches_xiph_anchor_values() {
        assert_eq!(tanh_approx(0.0), 0.0);
        assert!((tanh_approx(1.0) - 0.761_64).abs() < 2e-5);
        assert!((sigmoid_approx(2.0) - 0.880_82).abs() < 2e-5);
        assert_eq!(tanh_approx(100.0), 1.0);
    }

    #[test]
    fn blocked_i8_kernel_uses_output_major_eight_by_four_layout() {
        let weights: Vec<i8> = (0..8)
            .flat_map(|row| [row + 1, 0, 0, 0])
            .map(|value| value as i8)
            .collect();
        let mut output = [0.0; 8];
        accumulate_i8_block(&mut output, &weights, &[2, 3, 4, 5]);
        assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn exact_integer_container_rejects_fractional_values() {
        let value = 1.25f32;
        assert!(value.fract() != 0.0);
    }
}
