//! Native TEN-VAD v1.0 neural network primitive.
//!
//! Direct lowering of the official `v1.0-ONNX` graph: three separable
//! convolutions, two 64-unit ONNX LSTMs, and a `128 -> 32 -> 1` sigmoid
//! head. The LPCNet-derived streaming feature front-end is kept separate.

use vokra_core::{Result, VokraError};

use crate::fft::{FftPlan, RealFftPlan};

/// Feature frames consumed by one network step.
pub const CONTEXT_FRAMES: usize = 3;
/// Features produced per frontend frame.
pub const N_FEATURES: usize = 41;
/// Channel width of each separable convolution.
pub const CONV_CHANNELS: usize = 16;
/// Flattened input width of the first LSTM.
pub const LSTM0_INPUT: usize = 80;
/// Hidden width of both LSTM layers.
pub const HIDDEN_DIM: usize = 64;

/// One depthwise + pointwise convolution stage.
#[derive(Debug, Clone)]
pub struct SeparableConvWeights {
    /// Depthwise kernel in canonical ONNX order.
    pub depthwise: Vec<f32>,
    /// Pointwise kernel in canonical ONNX order.
    pub pointwise: Vec<f32>,
    /// Pointwise output bias.
    pub bias: Vec<f32>,
}

/// One ONNX LSTM layer (`iofc` gate order).
#[derive(Debug, Clone)]
pub struct LstmWeights {
    /// `[4 * hidden, input]`.
    pub weight_ih: Vec<f32>,
    /// `[4 * hidden, hidden]`.
    pub weight_hh: Vec<f32>,
    /// `[8 * hidden]`: input bias followed by recurrent bias.
    pub bias: Vec<f32>,
    /// Input width validated for this layer.
    pub input_size: usize,
}

/// Complete official TEN-VAD network weights.
#[derive(Debug, Clone)]
pub struct TenVadNetworkWeights {
    /// Initial two-dimensional separable convolution.
    pub conv0: SeparableConvWeights,
    /// First one-dimensional separable convolution.
    pub conv1: SeparableConvWeights,
    /// Second one-dimensional separable convolution.
    pub conv2: SeparableConvWeights,
    /// First 64-unit ONNX LSTM.
    pub lstm0: LstmWeights,
    /// Second 64-unit ONNX LSTM.
    pub lstm1: LstmWeights,
    /// TensorFlow layout `[128, 32]` (input-major).
    pub dense0_weight: Vec<f32>,
    /// Bias of the 128-to-32 dense layer.
    pub dense0_bias: Vec<f32>,
    /// TensorFlow layout `[32, 1]`.
    pub dense1_weight: Vec<f32>,
    /// Scalar output-layer bias.
    pub dense1_bias: f32,
}

impl TenVadNetworkWeights {
    /// Validates every fixed axis in the pinned v1.0 graph.
    pub fn validate(&self) -> Result<()> {
        check_len("conv0.depthwise", &self.conv0.depthwise, 9)?;
        check_len("conv0.pointwise", &self.conv0.pointwise, CONV_CHANNELS)?;
        check_len("conv0.bias", &self.conv0.bias, CONV_CHANNELS)?;
        for (label, stage) in [("conv1", &self.conv1), ("conv2", &self.conv2)] {
            check_len(
                &format!("{label}.depthwise"),
                &stage.depthwise,
                CONV_CHANNELS * 3,
            )?;
            check_len(
                &format!("{label}.pointwise"),
                &stage.pointwise,
                CONV_CHANNELS * CONV_CHANNELS,
            )?;
            check_len(&format!("{label}.bias"), &stage.bias, CONV_CHANNELS)?;
        }
        validate_lstm("lstm0", &self.lstm0, LSTM0_INPUT)?;
        validate_lstm("lstm1", &self.lstm1, HIDDEN_DIM)?;
        check_len("dense0.weight", &self.dense0_weight, HIDDEN_DIM * 2 * 32)?;
        check_len("dense0.bias", &self.dense0_bias, 32)?;
        check_len("dense1.weight", &self.dense1_weight, 32)?;
        if !self.dense1_bias.is_finite() {
            return Err(VokraError::InvalidArgument(
                "ten_vad network: dense1.bias is non-finite".to_owned(),
            ));
        }
        Ok(())
    }
}

fn check_len(label: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "ten_vad network: {label} has {} values, expected {expected}",
            values.len()
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "ten_vad network: {label} contains a non-finite value"
        )));
    }
    Ok(())
}

fn validate_lstm(label: &str, weights: &LstmWeights, expected_input: usize) -> Result<()> {
    if weights.input_size != expected_input {
        return Err(VokraError::InvalidArgument(format!(
            "ten_vad network: {label}.input_size is {}, expected {expected_input}",
            weights.input_size
        )));
    }
    check_len(
        &format!("{label}.weight_ih"),
        &weights.weight_ih,
        4 * HIDDEN_DIM * expected_input,
    )?;
    check_len(
        &format!("{label}.weight_hh"),
        &weights.weight_hh,
        4 * HIDDEN_DIM * HIDDEN_DIM,
    )?;
    check_len(&format!("{label}.bias"), &weights.bias, 8 * HIDDEN_DIM)
}

/// Four recurrent tensors exposed by the official ONNX graph.
#[derive(Debug, Clone, PartialEq)]
pub struct TenVadNetworkState {
    /// First-layer recurrent output state.
    pub hidden0: Vec<f32>,
    /// First-layer recurrent cell state.
    pub cell0: Vec<f32>,
    /// Second-layer recurrent output state.
    pub hidden1: Vec<f32>,
    /// Second-layer recurrent cell state.
    pub cell1: Vec<f32>,
}

impl Default for TenVadNetworkState {
    fn default() -> Self {
        Self {
            hidden0: vec![0.0; HIDDEN_DIM],
            cell0: vec![0.0; HIDDEN_DIM],
            hidden1: vec![0.0; HIDDEN_DIM],
            cell1: vec![0.0; HIDDEN_DIM],
        }
    }
}

impl TenVadNetworkState {
    /// Clears all four recurrent tensors.
    pub fn reset(&mut self) {
        self.hidden0.fill(0.0);
        self.cell0.fill(0.0);
        self.hidden1.fill(0.0);
        self.cell1.fill(0.0);
    }
}

/// Runs one `3 x 41` feature context and advances both LSTM layers.
pub fn network_forward(
    features: &[f32],
    weights: &TenVadNetworkWeights,
    state: &mut TenVadNetworkState,
) -> Result<f32> {
    if features.len() != CONTEXT_FRAMES * N_FEATURES {
        return Err(VokraError::InvalidArgument(format!(
            "ten_vad network: features have {} values, expected {} (3 x 41)",
            features.len(),
            CONTEXT_FRAMES * N_FEATURES
        )));
    }
    if features.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "ten_vad network: features contain a non-finite value".to_owned(),
        ));
    }
    weights.validate()?;
    for (label, values) in [
        ("hidden0", &state.hidden0),
        ("cell0", &state.cell0),
        ("hidden1", &state.hidden1),
        ("cell1", &state.cell1),
    ] {
        check_len(label, values, HIDDEN_DIM)?;
    }

    // [1,3,41] -> valid 3x3 -> [1,1,39] -> pointwise 16 -> pool -> [16,19].
    let mut conv0_scalar = [0.0f32; 39];
    for x in 0..39 {
        let mut sum = 0.0f32;
        for y in 0..3 {
            for kernel in 0..3 {
                sum +=
                    features[y * N_FEATURES + x + kernel] * weights.conv0.depthwise[y * 3 + kernel];
            }
        }
        conv0_scalar[x] = sum;
    }
    let mut conv0 = vec![0.0f32; CONV_CHANNELS * 39];
    for channel in 0..CONV_CHANNELS {
        let scale = weights.conv0.pointwise[channel];
        let bias = weights.conv0.bias[channel];
        for x in 0..39 {
            conv0[channel * 39 + x] = (conv0_scalar[x] * scale + bias).max(0.0);
        }
    }
    let mut pooled = vec![0.0f32; CONV_CHANNELS * 19];
    for channel in 0..CONV_CHANNELS {
        for x in 0..19 {
            let start = x * 2;
            pooled[channel * 19 + x] = conv0[channel * 39 + start]
                .max(conv0[channel * 39 + start + 1])
                .max(conv0[channel * 39 + start + 2]);
        }
    }

    let conv1 = separable_1d(&pooled, 19, 10, 2, 1, &weights.conv1);
    let conv2 = separable_1d(&conv1, 10, 5, 2, 0, &weights.conv2);

    // NCHW [16,1,5] -> NHWC [5,16] -> one 80-wide LSTM timestep.
    let mut lstm_input = [0.0f32; LSTM0_INPUT];
    for time in 0..5 {
        for channel in 0..CONV_CHANNELS {
            lstm_input[time * CONV_CHANNELS + channel] = conv2[channel * 5 + time];
        }
    }
    let hidden0 = lstm_step(
        &lstm_input,
        &weights.lstm0,
        &mut state.hidden0,
        &mut state.cell0,
    );
    let hidden1 = lstm_step(
        &hidden0,
        &weights.lstm1,
        &mut state.hidden1,
        &mut state.cell1,
    );

    // Official concat order is layer-2 output followed by layer-1 output.
    let mut dense_input = Vec::with_capacity(HIDDEN_DIM * 2);
    dense_input.extend_from_slice(&hidden1);
    dense_input.extend_from_slice(&hidden0);
    let mut dense = [0.0f32; 32];
    for (output, dense_value) in dense.iter_mut().enumerate() {
        let mut sum = weights.dense0_bias[output];
        for (input, &input_value) in dense_input.iter().enumerate() {
            sum += input_value * weights.dense0_weight[input * 32 + output];
        }
        *dense_value = sum.max(0.0);
    }
    let logit = dense
        .iter()
        .zip(&weights.dense1_weight)
        .fold(weights.dense1_bias, |sum, (&value, &weight)| {
            sum + value * weight
        });
    Ok(sigmoid(logit))
}

fn separable_1d(
    input: &[f32],
    input_width: usize,
    output_width: usize,
    stride: usize,
    pad_left: usize,
    weights: &SeparableConvWeights,
) -> Vec<f32> {
    let mut depthwise = vec![0.0f32; CONV_CHANNELS * output_width];
    for channel in 0..CONV_CHANNELS {
        for output in 0..output_width {
            let base = output * stride;
            let mut sum = 0.0f32;
            for kernel in 0..3 {
                let padded = base + kernel;
                if padded >= pad_left {
                    let input_x = padded - pad_left;
                    if input_x < input_width {
                        sum += input[channel * input_width + input_x]
                            * weights.depthwise[channel * 3 + kernel];
                    }
                }
            }
            depthwise[channel * output_width + output] = sum;
        }
    }
    let mut result = vec![0.0f32; CONV_CHANNELS * output_width];
    for output_channel in 0..CONV_CHANNELS {
        for x in 0..output_width {
            let mut sum = weights.bias[output_channel];
            for input_channel in 0..CONV_CHANNELS {
                sum += depthwise[input_channel * output_width + x]
                    * weights.pointwise[output_channel * CONV_CHANNELS + input_channel];
            }
            result[output_channel * output_width + x] = sum.max(0.0);
        }
    }
    result
}

fn lstm_step(
    input: &[f32],
    weights: &LstmWeights,
    hidden: &mut [f32],
    cell: &mut [f32],
) -> Vec<f32> {
    let previous_hidden = hidden.to_vec();
    let previous_cell = cell.to_vec();
    let mut gates = [0.0f32; 4 * HIDDEN_DIM];
    for gate in 0..4 {
        for unit in 0..HIDDEN_DIM {
            let row = gate * HIDDEN_DIM + unit;
            let mut sum = weights.bias[row] + weights.bias[4 * HIDDEN_DIM + row];
            for (column, &value) in input.iter().enumerate() {
                sum += value * weights.weight_ih[row * weights.input_size + column];
            }
            for (column, &value) in previous_hidden.iter().enumerate() {
                sum += value * weights.weight_hh[row * HIDDEN_DIM + column];
            }
            gates[row] = sum;
        }
    }
    for unit in 0..HIDDEN_DIM {
        // ONNX gate order: input, output, forget, cell.
        let input_gate = sigmoid(gates[unit]);
        let output_gate = sigmoid(gates[HIDDEN_DIM + unit]);
        let forget_gate = sigmoid(gates[2 * HIDDEN_DIM + unit]);
        let cell_gate = gates[3 * HIDDEN_DIM + unit].tanh();
        cell[unit] = forget_gate * previous_cell[unit] + input_gate * cell_gate;
        hidden[unit] = output_gate * cell[unit].tanh();
    }
    hidden.to_vec()
}

#[inline]
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

// ---------------------------------------------------------------------------
// Official LPCNet-derived streaming frontend (BSD-2-Clause/BSD-3-Clause).
// ---------------------------------------------------------------------------

/// Required input PCM sample rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// Samples consumed by one frontend step.
pub const HOP_SIZE: usize = 256;
/// Internal STFT transform size.
pub const FFT_SIZE: usize = 1024;
/// Periodic Hann window size.
pub const WINDOW_SIZE: usize = 768;
const N_BINS: usize = FFT_SIZE / 2 + 1;
const N_MELS: usize = 40;
const EPSILON: f32 = 1.0e-20;

// Preserve the decimal spellings shipped in the upstream C coefficient file:
// Rust rounds each literal to f32, while shortening them obscures provenance.
#[allow(clippy::excessive_precision)]
const FEATURE_MEANS: [f32; N_FEATURES] = [
    -8.198236465454,
    -6.265716552734,
    -5.483818531036,
    -4.758691310883,
    -4.417088985443,
    -4.142892837524,
    -3.912850379944,
    -3.845927953720,
    -3.657090425491,
    -3.723418712616,
    -3.876134157181,
    -3.843890905380,
    -3.690405130386,
    -3.75606584549,
    -3.698696136475,
    -3.650463104248,
    -3.70046877861,
    -3.567321300507,
    -3.498900175095,
    -3.477807044983,
    -3.458816051483,
    -3.444923877716,
    -3.40132856369,
    -3.306261301041,
    -3.27855682373,
    -3.2332508564,
    -3.198616027832,
    -3.204526424408,
    -3.208798646927,
    -3.257838010788,
    -3.381376743317,
    -3.534021377563,
    -3.640867948532,
    -3.726858854294,
    -3.773730993271,
    -3.804667234421,
    -3.832901000977,
    -3.871120452881,
    -3.990592956543,
    -4.480289459229,
    92.35690307617,
];

#[allow(clippy::excessive_precision)]
const FEATURE_STDS: [f32; N_FEATURES] = [
    5.166063785553,
    4.977209568024,
    4.698895931244,
    4.630621433258,
    4.634347915649,
    4.641156196594,
    4.640676498413,
    4.666367053986,
    4.650534629822,
    4.640020847321,
    4.637400150299,
    4.620099067688,
    4.596316337585,
    4.562654972076,
    4.554360389709,
    4.566910743713,
    4.56248998642,
    4.5624127388,
    4.585299491882,
    4.600179672241,
    4.592845916748,
    4.585922718048,
    4.583496570587,
    4.626092910767,
    4.626957893372,
    4.626289367676,
    4.637005805969,
    4.683015823364,
    4.726813793182,
    4.734289646149,
    4.753227233887,
    4.849722862244,
    4.869434833527,
    4.884482860565,
    4.921327114105,
    4.959212303162,
    4.996619224548,
    5.044823646545,
    5.07221698761,
    5.096439361572,
    115.2136917114,
];

const BIQUAD_B: [[f32; 3]; 5] = [
    [1.0, 1.198825, 1.0],
    [1.0, -0.5674614, 1.0],
    [1.0, -1.099061, 1.0],
    [1.0, -1.265846, 1.0],
    [1.0, -1.318849, 1.0],
];
const BIQUAD_A: [[f32; 3]; 5] = [
    [1.0, -1.445267, 0.5463974],
    [1.0, -1.42672, 0.6820138],
    [1.0, -1.408255, 0.8286664],
    [1.0, -1.400909, 0.924032],
    [1.0, -1.408242, 0.9789776],
];
const BIQUAD_G: f32 = 0.2692541;

const PITCH_BAND_START: [i32; 18] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40,
];
const PITCH_BAND_COMP: [f32; 18] = [
    0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.666667, 0.5, 0.5, 0.5, 0.333333, 0.25, 0.25, 0.2,
    0.166667, 0.173913,
];

/// Stateful official TEN-VAD PCM-to-`3 x 41` feature frontend.
///
/// Input is mono normalized `f32` PCM. Samples are multiplied by 32768 before
/// the upstream equations, matching the public C wrapper's `int16_t -> float`
/// conversion exactly for WAV-derived samples.
pub struct TenVadFrontend {
    forward_fft: FftPlan,
    fft: RealFftPlan,
    stft_queue: Vec<f32>,
    feature_context: Vec<f32>,
    previous_sample: f32,
    mel_filters: Vec<f32>,
    pitch: PitchState,
}

impl Default for TenVadFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl TenVadFrontend {
    /// Creates a zeroed streaming frontend for 16 kHz mono PCM.
    #[must_use]
    pub fn new() -> Self {
        Self {
            forward_fft: FftPlan::new(FFT_SIZE),
            fft: RealFftPlan::new(FFT_SIZE),
            stft_queue: vec![0.0; WINDOW_SIZE],
            feature_context: vec![0.0; CONTEXT_FRAMES * N_FEATURES],
            previous_sample: 0.0,
            mel_filters: make_mel_filters(),
            pitch: PitchState::new(),
        }
    }

    /// Advances one 256-sample frame and returns the complete 3-frame context.
    pub fn process_frame(&mut self, pcm: &[f32]) -> Result<&[f32]> {
        if pcm.len() != HOP_SIZE {
            return Err(VokraError::InvalidArgument(format!(
                "ten_vad frontend: frame has {} samples, expected {HOP_SIZE}",
                pcm.len()
            )));
        }
        if pcm.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "ten_vad frontend: PCM contains a non-finite sample".to_owned(),
            ));
        }
        let raw = pcm.iter().map(|value| value * 32768.0).collect::<Vec<_>>();
        self.stft_queue.copy_within(HOP_SIZE.., 0);
        for (index, &sample) in raw.iter().enumerate() {
            let emphasized = sample - 0.97 * self.previous_sample;
            self.previous_sample = sample;
            self.stft_queue[WINDOW_SIZE - HOP_SIZE + index] = emphasized;
        }
        let mut fft_input = vec![0.0f32; FFT_SIZE];
        for (index, output) in fft_input.iter_mut().take(WINDOW_SIZE).enumerate() {
            // `coeff.h` stores the double-precision periodic Hann at `%.7e`
            // (eight significant decimal digits), then C parses each literal
            // to f32. Reproduce both rounding stages; a plain f32 cosine
            // drifts by two ulps and a direct f64->f32 cast by one.
            let phase = 2.0 * core::f64::consts::PI * index as f64 / WINDOW_SIZE as f64;
            let ideal = 0.5 - 0.5 * phase.cos();
            let window = if ideal == 0.0 {
                0.0
            } else {
                let decimal_exponent = ideal.abs().log10().floor() as i32;
                let step = 10.0f64.powi(decimal_exponent - 7);
                ((ideal / step).round() * step) as f32
            };
            *output = self.stft_queue[index] * window;
        }
        let spectrum = official_scaled_fft(&self.forward_fft, &fft_input);
        let power = spectrum
            .iter()
            .map(|value| value.re.mul_add(value.re, value.im * value.im))
            .collect::<Vec<_>>();
        let pitch_hz = self.pitch.process(&raw, &power, &self.fft);

        self.feature_context.copy_within(N_FEATURES.., 0);
        let current = &mut self.feature_context[(CONTEXT_FRAMES - 1) * N_FEATURES..];
        for mel in 0..N_MELS {
            let filter = &self.mel_filters[mel * N_BINS..(mel + 1) * N_BINS];
            let energy = power
                .iter()
                .zip(filter)
                .fold(0.0f32, |sum, (&value, &coefficient)| {
                    value.mul_add(coefficient, sum)
                })
                / (32768.0 * 32768.0);
            let logged = (energy + EPSILON).ln();
            current[mel] = (logged - FEATURE_MEANS[mel]) / (FEATURE_STDS[mel] + EPSILON);
        }
        current[N_MELS] = (pitch_hz - FEATURE_MEANS[N_MELS]) / (FEATURE_STDS[N_MELS] + EPSILON);
        Ok(&self.feature_context)
    }

    /// Clears STFT, feature-context, pre-emphasis, and pitch state.
    pub fn reset(&mut self) {
        self.stft_queue.fill(0.0);
        self.feature_context.fill(0.0);
        self.previous_sample = 0.0;
        self.pitch.reset();
    }
}

fn make_mel_filters() -> Vec<f32> {
    let mut bins = [0usize; N_MELS + 2];
    let low_mel = 2595.0f32 * (1.0f32 + 0.0f32 / 700.0).log10();
    let high_mel = 2595.0f32 * (1.0f32 + 8000.0f32 / 700.0).log10();
    for (index, bin) in bins.iter_mut().enumerate() {
        let mel = index as f32 * (high_mel - low_mel) / (N_MELS as f32 + 1.0) + low_mel;
        let hz = 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);
        *bin = ((FFT_SIZE as f32 + 1.0) * hz / SAMPLE_RATE as f32) as usize;
    }
    let mut filters = vec![0.0f32; N_MELS * N_BINS];
    for band in 0..N_MELS {
        for bin in bins[band]..bins[band + 1] {
            filters[band * N_BINS + bin] =
                (bin - bins[band]) as f32 / (bins[band + 1] - bins[band]) as f32;
        }
        for bin in bins[band + 1]..bins[band + 2] {
            filters[band * N_BINS + bin] =
                (bins[band + 2] - bin) as f32 / (bins[band + 2] - bins[band + 1]) as f32;
        }
    }
    filters
}

/// Mirrors the official wrapper's exact binary input pre-scale and output
/// re-scale around its f32 FFT. Keeping the intermediate magnitudes near one
/// is observable after the log-mel reduction, even though the two factors
/// cancel algebraically.
fn official_scaled_fft(plan: &FftPlan, input: &[f32]) -> Vec<vokra_core::Complex32> {
    debug_assert_eq!(input.len(), FFT_SIZE);
    let scaled = input
        .iter()
        .map(|&value| vokra_core::Complex32::from_real(value * (1.0 / FFT_SIZE as f32)))
        .collect::<Vec<_>>();
    plan.forward_raw(&scaled)
        .into_iter()
        .take(N_BINS)
        .map(|value| value.scale(FFT_SIZE as f32))
        .collect()
}

struct PitchState {
    input_queue: Vec<f32>,
    lpc: [f32; 16],
    pitch_memory: [f32; 16],
    pitch_filter: f32,
    biquad_state: [[f32; 2]; 5],
    excitation: Vec<f32>,
    xcorr_offset: usize,
    xcorr: Vec<Vec<f32>>,
    frame_weight: [f32; 6],
    pitch_path: [Vec<f32>; 2],
    pitch_previous: Vec<Vec<usize>>,
    pitch_path_max: f32,
    best_period: usize,
    dct: [[f32; 18]; 18],
}

impl PitchState {
    // This spelling is copied from upstream's DCT initializer and is kept
    // distinct from Rust's full-precision PI for C-ABI feature parity.
    #[allow(clippy::approx_constant)]
    fn new() -> Self {
        let mut dct = [[0.0f32; 18]; 18];
        for (row, values) in dct.iter_mut().enumerate() {
            for (column, value) in values.iter_mut().enumerate() {
                *value = ((row as f32 + 0.5) * column as f32 * 3.1415926 / 18.0).cos();
                if column == 0 {
                    *value *= 0.5f32.sqrt();
                }
            }
        }
        Self {
            input_queue: vec![0.0; HOP_SIZE * 2],
            lpc: [0.0; 16],
            pitch_memory: [0.0; 16],
            pitch_filter: 0.0,
            biquad_state: [[0.0; 2]; 5],
            excitation: vec![0.0; 64 + 64 + 1],
            xcorr_offset: 0,
            xcorr: vec![vec![0.0; 65]; 6],
            frame_weight: [0.0; 6],
            pitch_path: [vec![0.0; 64], vec![0.0; 64]],
            pitch_previous: vec![vec![0; 64]; 6],
            pitch_path_max: 0.0,
            best_period: 0,
            dct,
        }
    }

    fn reset(&mut self) {
        self.input_queue.fill(0.0);
        self.lpc.fill(0.0);
        self.pitch_memory.fill(0.0);
        self.pitch_filter = 0.0;
        self.biquad_state.fill([0.0; 2]);
        self.excitation.fill(0.0);
        self.xcorr_offset = 0;
        for values in &mut self.xcorr {
            values.fill(0.0);
        }
        self.frame_weight.fill(0.0);
        self.pitch_path[0].fill(0.0);
        self.pitch_path[1].fill(0.0);
        for values in &mut self.pitch_previous {
            values.fill(0);
        }
        self.pitch_path_max = 0.0;
        self.best_period = 0;
    }

    fn process(&mut self, frame: &[f32], bin_power: &[f32], fft: &RealFftPlan) -> f32 {
        let band_power = pitch_band_energy(bin_power);
        let mut log_bands = [0.0f32; 18];
        let mut log_max = -2.0f32;
        let mut follow = -2.0f32;
        for index in 0..18 {
            let value = (0.01 + band_power[index]).log10();
            log_bands[index] = value.max(log_max - 8.0).max(follow - 2.5);
            log_max = log_max.max(log_bands[index]);
            follow = (follow - 2.5).max(log_bands[index]);
        }
        let cepstrum = dct_forward(&self.dct, &log_bands);
        self.lpc = lpc_from_cepstrum(&self.dct, &cepstrum, fft);

        self.input_queue.copy_within(HOP_SIZE.., 0);
        self.input_queue[HOP_SIZE..].copy_from_slice(frame);
        let aligned = self.input_queue[176..176 + HOP_SIZE].to_vec();
        let mut filtered = [0.0f32; HOP_SIZE];
        for (index, &sample) in aligned.iter().enumerate() {
            let mut prediction = sample;
            for order in 0..16 {
                prediction += self.lpc[order] * self.pitch_memory[order];
            }
            self.pitch_memory.copy_within(..15, 1);
            self.pitch_memory[0] = sample;
            filtered[index] = prediction + 0.7 * self.pitch_filter;
            self.pitch_filter = prediction;
        }
        let low_passed = self.biquad(&filtered);
        let downsampled = low_passed.iter().step_by(4).copied().collect::<Vec<_>>();
        self.excitation.copy_within(downsampled.len().., 0);
        let start = self.excitation.len() - downsampled.len();
        self.excitation[start..].copy_from_slice(&downsampled);
        self.track_pitch()
    }

    fn biquad(&mut self, input: &[f32]) -> Vec<f32> {
        let mut current = input.to_vec();
        for section in 0..5 {
            let mut output = vec![0.0f32; current.len()];
            for (index, &sample) in current.iter().enumerate() {
                let value = sample
                    - BIQUAD_A[section][1] * self.biquad_state[section][0]
                    - BIQUAD_A[section][2] * self.biquad_state[section][1];
                output[index] = BIQUAD_G
                    * (BIQUAD_B[section][0] * value
                        + BIQUAD_B[section][1] * self.biquad_state[section][0]
                        + BIQUAD_B[section][2] * self.biquad_state[section][1]);
                self.biquad_state[section][1] = self.biquad_state[section][0];
                self.biquad_state[section][0] = value;
            }
            current = output;
        }
        current
    }

    fn track_pitch(&mut self) -> f32 {
        const MAX_PERIOD: usize = 64;
        const MIN_PERIOD: usize = 8;
        const HALF_HOP: usize = 32;
        let excitation_sq = self
            .excitation
            .iter()
            .map(|value| value * value)
            .collect::<Vec<_>>();
        self.frame_weight.copy_within(2.., 0);
        for sub in 0..2 {
            let xcorr_index = 2 * self.xcorr_offset + sub;
            let offset = sub * HALF_HOP;
            let reference = &self.excitation[MAX_PERIOD + offset..MAX_PERIOD + offset + HALF_HOP];
            let energy0 = reference.iter().map(|value| value * value).sum::<f32>();
            self.frame_weight[4 + sub] = energy0;
            let mut shifted_energy = excitation_sq[offset..offset + HALF_HOP].iter().sum::<f32>();
            for lag in 0..MAX_PERIOD {
                if lag > 0 {
                    shifted_energy = (shifted_energy - excitation_sq[offset + lag - 1]).max(0.0)
                        + excitation_sq[offset + lag + HALF_HOP - 1];
                }
                let correlation = reference
                    .iter()
                    .zip(&self.excitation[offset + lag..offset + lag + HALF_HOP])
                    .fold(0.0f32, |sum, (&left, &right)| sum + left * right);
                let denominator = (shifted_energy + 1.0 + energy0).max(1.0e-12);
                self.xcorr[xcorr_index][lag] = 2.0 * correlation / denominator;
            }
            for lag in 0..MAX_PERIOD - 2 * MIN_PERIOD {
                let half0 = (MAX_PERIOD + lag) / 2;
                let half1 = (MAX_PERIOD + lag + 2) / 2;
                let half2 = (MAX_PERIOD + lag - 1) / 2;
                let competing = self.xcorr[xcorr_index][half0]
                    .max(self.xcorr[xcorr_index][half1])
                    .max(self.xcorr[xcorr_index][half2]);
                if self.xcorr[xcorr_index][lag] < competing * 1.1 {
                    self.xcorr[xcorr_index][lag] *= 0.8;
                }
            }
        }
        self.xcorr_offset = (self.xcorr_offset + 1) % 3;

        let weight_sum = self.frame_weight.iter().sum::<f32>() + 1.0e-15;
        let normalized_weight = self
            .frame_weight
            .map(|value| value * (self.frame_weight.len() as f32 / weight_sum));
        let xcorr_snapshot = self.xcorr.clone();
        for index in 0..4 {
            self.pitch_previous[index] = self.pitch_previous[index + 2].clone();
        }
        for (sub, &sub_weight) in normalized_weight.iter().enumerate().skip(4) {
            let xcorr_index = (sub + self.xcorr_offset * 2) % 6;
            for (period, &xcorr_value) in xcorr_snapshot[xcorr_index]
                .iter()
                .enumerate()
                .take(MAX_PERIOD - MIN_PERIOD)
            {
                let mut maximum = self.pitch_path_max - 1.0e10;
                let mut previous = self.best_period;
                let lower = 0i32.min(4 - period as i32);
                for delta in lower..=4 {
                    let candidate = period as i32 + delta;
                    if candidate < 0 || candidate >= (MAX_PERIOD - MIN_PERIOD) as i32 {
                        continue;
                    }
                    let score = self.pitch_path[0][candidate as usize]
                        - 0.02 * (delta.abs() * delta.abs()) as f32;
                    if score > maximum {
                        maximum = score;
                        previous = candidate as usize;
                    }
                }
                self.pitch_previous[sub][period] = previous;
                self.pitch_path[1][period] = maximum + sub_weight * xcorr_value;
            }
            let mut maximum = -1.0e15f32;
            let mut best = 0usize;
            for period in 0..MAX_PERIOD - MIN_PERIOD {
                if self.pitch_path[1][period] > maximum {
                    maximum = self.pitch_path[1][period];
                    best = period;
                }
            }
            self.pitch_path_max = maximum;
            self.best_period = best;
            let updated_path = self.pitch_path[1].clone();
            self.pitch_path[0].copy_from_slice(&updated_path);
            for value in &mut self.pitch_path[0][..MAX_PERIOD - MIN_PERIOD] {
                *value -= maximum;
            }
        }

        let mut local_periods = [0usize; 6];
        let mut period = self.best_period;
        let mut frame_correlation = 0.0f32;
        for sub in (0..6).rev() {
            local_periods[sub] = MAX_PERIOD - period;
            let xcorr_index = (sub + self.xcorr_offset * 2) % 6;
            frame_correlation += normalized_weight[sub] * xcorr_snapshot[xcorr_index][period];
            period = self.pitch_previous[sub][period];
        }
        frame_correlation = (frame_correlation / 6.0).max(0.0);
        let voiced = frame_correlation >= 0.4;
        let mut sw = 0.0f32;
        let mut sx = 0.0f32;
        let mut sxx = 0.0f32;
        let mut sxy = 0.0f32;
        let mut sy = 0.0f32;
        for sub in 0..6 {
            let weight = normalized_weight[sub];
            let x = sub as f32;
            let y = local_periods[sub] as f32;
            sw += weight;
            sx += weight * x;
            sxx += weight * x * x;
            sxy += weight * x * y;
            sy += weight * y;
        }
        let denominator = sw * sxx - sx * sx;
        let mut slope = if denominator == 0.0 {
            (sw * sxy - sx * sy) / 1.0e-15
        } else {
            (sw * sxy - sx * sy) / denominator
        };
        if voiced {
            let bound = (sy / sw) / 24.0;
            slope = slope.clamp(-bound, bound);
        } else {
            slope = 0.0;
        }
        let intercept = (sy - slope * sx) / sw;
        let estimated_period = intercept + 5.5 * slope;
        if voiced {
            4000.0 / estimated_period.max(1.0)
        } else {
            0.0
        }
    }
}

fn pitch_band_energy(power: &[f32]) -> [f32; 18] {
    let mut bands = [0.0f32; 18];
    let scale = FFT_SIZE as f32 / 80.0;
    for band in 0..17 {
        let width =
            ((PITCH_BAND_START[band + 1] - PITCH_BAND_START[band]) as f32 * scale).round() as usize;
        let offset = (PITCH_BAND_START[band] as f32 * scale).round() as usize;
        for index in 0..width {
            let fraction = index as f32 / width as f32;
            let source = (offset + index).min(N_BINS - 1);
            bands[band] += (1.0 - fraction) * power[source];
            bands[band + 1] += fraction * power[source];
        }
    }
    bands[0] *= 2.0;
    bands[17] *= 2.0;
    bands
}

fn dct_forward(table: &[[f32; 18]; 18], input: &[f32; 18]) -> [f32; 18] {
    let mut output = [0.0f32; 18];
    let scale = (2.0f32 / 18.0).sqrt();
    for (column, value) in output.iter_mut().enumerate() {
        for row in 0..18 {
            *value += input[row] * table[row][column];
        }
        *value *= scale;
    }
    output
}

fn dct_inverse(table: &[[f32; 18]; 18], input: &[f32; 18]) -> [f32; 18] {
    let mut output = [0.0f32; 18];
    let scale = (2.0f32 / 18.0).sqrt();
    for (row, value) in output.iter_mut().enumerate() {
        for column in 0..18 {
            *value += input[column] * table[row][column];
        }
        *value *= scale;
    }
    output
}

fn lpc_from_cepstrum(
    table: &[[f32; 18]; 18],
    cepstrum: &[f32; 18],
    fft: &RealFftPlan,
) -> [f32; 16] {
    let mut band_energy = dct_inverse(table, cepstrum);
    for index in 0..18 {
        band_energy[index] = 10.0f32.powf(band_energy[index]) * PITCH_BAND_COMP[index];
    }
    let mut spectrum = vec![vokra_core::Complex32::ZERO; N_BINS];
    let scale = FFT_SIZE as f32 / 80.0;
    for band in 0..17 {
        let width =
            ((PITCH_BAND_START[band + 1] - PITCH_BAND_START[band]) as f32 * scale).round() as usize;
        let offset = (PITCH_BAND_START[band] as f32 * scale).round() as usize;
        for index in 0..width {
            let fraction = index as f32 / width as f32;
            let target = (offset + index).min(N_BINS - 1);
            spectrum[target].re =
                (1.0 - fraction) * band_energy[band] + fraction * band_energy[band + 1];
        }
    }
    spectrum[N_BINS - 1].re = 0.0;
    let autocorrelation_full = fft.inverse(&spectrum);
    let mut autocorrelation = [0.0f32; 17];
    autocorrelation.copy_from_slice(&autocorrelation_full[..17]);
    autocorrelation[0] += autocorrelation[0] * 1.0e-4 + WINDOW_SIZE as f32 / 12.0 / 38.0;
    for (lag, value) in autocorrelation.iter_mut().enumerate().skip(1) {
        *value *= 1.0 - 6.0e-5 * (lag * lag) as f32;
    }
    levinson_durbin(&autocorrelation)
}

fn levinson_durbin(autocorrelation: &[f32; 17]) -> [f32; 16] {
    let mut lpc = [0.0f32; 16];
    let mut error = autocorrelation[0];
    if error == 0.0 {
        return lpc;
    }
    for order in 0..16 {
        let mut residual = autocorrelation[order + 1];
        for previous in 0..order {
            residual += lpc[previous] * autocorrelation[order - previous];
        }
        let reflection = -residual / error;
        lpc[order] = reflection;
        for index in 0..order.div_ceil(2) {
            let left = lpc[index];
            let right = lpc[order - 1 - index];
            lpc[index] = left + reflection * right;
            lpc[order - 1 - index] = right + reflection * left;
        }
        error -= reflection * reflection * error;
        if error < 0.001 * autocorrelation[0] {
            break;
        }
    }
    lpc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_weights() -> TenVadNetworkWeights {
        let stage0 = SeparableConvWeights {
            depthwise: vec![0.0; 9],
            pointwise: vec![0.0; CONV_CHANNELS],
            bias: vec![0.0; CONV_CHANNELS],
        };
        let stage = SeparableConvWeights {
            depthwise: vec![0.0; CONV_CHANNELS * 3],
            pointwise: vec![0.0; CONV_CHANNELS * CONV_CHANNELS],
            bias: vec![0.0; CONV_CHANNELS],
        };
        TenVadNetworkWeights {
            conv0: stage0,
            conv1: stage.clone(),
            conv2: stage,
            lstm0: LstmWeights {
                weight_ih: vec![0.0; 4 * HIDDEN_DIM * LSTM0_INPUT],
                weight_hh: vec![0.0; 4 * HIDDEN_DIM * HIDDEN_DIM],
                bias: vec![0.0; 8 * HIDDEN_DIM],
                input_size: LSTM0_INPUT,
            },
            lstm1: LstmWeights {
                weight_ih: vec![0.0; 4 * HIDDEN_DIM * HIDDEN_DIM],
                weight_hh: vec![0.0; 4 * HIDDEN_DIM * HIDDEN_DIM],
                bias: vec![0.0; 8 * HIDDEN_DIM],
                input_size: HIDDEN_DIM,
            },
            dense0_weight: vec![0.0; HIDDEN_DIM * 2 * 32],
            dense0_bias: vec![0.0; 32],
            dense1_weight: vec![0.0; 32],
            dense1_bias: 0.0,
        }
    }

    #[test]
    fn zero_graph_emits_half_and_preserves_zero_state() {
        let weights = zero_weights();
        let mut state = TenVadNetworkState::default();
        let probability = network_forward(&[0.0; 123], &weights, &mut state).unwrap();
        assert_eq!(probability, 0.5);
        assert_eq!(state, TenVadNetworkState::default());
    }

    #[test]
    fn rejects_wrong_feature_shape() {
        let error = network_forward(
            &[0.0; 122],
            &zero_weights(),
            &mut TenVadNetworkState::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("expected 123"));
    }

    #[test]
    fn frontend_rejects_wrong_hop_and_reset_is_deterministic() {
        let mut frontend = TenVadFrontend::new();
        let error = frontend.process_frame(&[0.0; HOP_SIZE - 1]).unwrap_err();
        assert!(error.to_string().contains("expected 256"));

        let frame = (0..HOP_SIZE)
            .map(|index| (index as f32 * 0.031).sin() * 0.1)
            .collect::<Vec<_>>();
        let first = frontend.process_frame(&frame).unwrap().to_vec();
        frontend.process_frame(&frame).unwrap();
        frontend.reset();
        let after_reset = frontend.process_frame(&frame).unwrap().to_vec();
        assert_eq!(first, after_reset);
    }
}
