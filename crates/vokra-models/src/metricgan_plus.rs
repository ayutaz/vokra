//! Native SpeechBrain MetricGAN+ VoiceBank speech-enhancement runtime.
//!
//! The public `vokra/metricgan-plus-voicebank` GGUF contains the exact 21
//! tensors of SpeechBrain's [`EnhancementGenerator`]: a two-layer
//! bidirectional LSTM, `400 -> 300 -> 257` mask head and a learned 257-bin
//! sigmoid slope.  This module binds that released manifest strictly and
//! reproduces the upstream inference pipeline:
//!
//! ```text
//! 16 kHz mono PCM
//!   -> centered 512-point periodic-Hamming STFT (256-sample hop)
//!   -> log1p(sqrt(power + 1e-14))
//!   -> BiLSTM(257, 200, layers=2, bidirectional)
//!   -> Linear(400, 300) + LeakyReLU(0.3)
//!   -> Linear(300, 257) + 1.2 * sigmoid(slope * x)
//!   -> mask the log-magnitude, expm1, reuse noisy phase
//!   -> matching iSTFT and peak normalization
//! ```
//!
//! Primary sources:
//!
//! - `speechbrain/lobes/models/MetricGAN.py::EnhancementGenerator`
//! - `speechbrain/inference/enhancement.py::SpectralMaskEnhancement`
//! - `speechbrain/processing/features.py::{STFT, ISTFT, spectral_magnitude}`
//! - `speechbrain/processing/signal_processing.py::resynthesize`
//! - `speechbrain/metricgan-plus-voicebank/hyperparams.yaml`
//!
//! CPU deliberately preserves a scalar oracle.  For any non-CPU selection,
//! every learned input/recurrent projection and both dense layers execute
//! through one [`Compute`] selected up front.  STFT/iSTFT, recurrent gate
//! state, activations, phase assembly and peak normalization are host DSP/glue;
//! they are not a hidden CPU inference fallback.  An unavailable backend or
//! uncovered learned op is returned before the frontend runs (FR-EX-08).

use std::collections::BTreeSet;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile, chunks};
use vokra_core::ir::graph::{
    IstftAttrs, Normalization, PadMode, StftAttrs, Window, WindowSymmetry,
};
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::{Spectrogram, istft, stft};

use crate::compute::{Compute, HotOp};

/// Architecture tag written by the MetricGAN+ converter.
pub const ARCH: &str = "metricgan_plus";
/// Exact released model id.
pub const NAME: &str = "metricgan-plus-voicebank";
/// Model category written by the converter.
pub const CATEGORY: &str = "enhancement";
/// Official upstream Hugging Face repository.
pub const UPSTREAM_HF: &str = "speechbrain/metricgan-plus-voicebank";
/// Custom metadata key used by the pass-through converter.
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// Custom upstream-repository metadata key used by the converter.
pub const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Required PCM sample rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// SpeechBrain FFT size and window length.
pub const N_FFT: usize = 512;
/// SpeechBrain 16 ms hop at 16 kHz.
pub const HOP_LENGTH: usize = 256;
/// One-sided frequency-bin count.
pub const N_BINS: usize = N_FFT / 2 + 1;
/// Per-direction LSTM hidden width.
pub const HIDDEN_SIZE: usize = 200;
/// Bidirectional LSTM layer count.
pub const NUM_LAYERS: usize = 2;
/// First mask-head hidden width.
pub const MASK_HIDDEN_SIZE: usize = 300;
/// Exact released tensor count.
pub const TENSOR_COUNT: usize = 21;

/// Every learned reduction required by the complete Metal path.
pub const METRICGAN_PLUS_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::Gemv];

const MAGNITUDE_EPS: f32 = 1.0e-14;
const NORMALIZE_EPS: f32 = 1.0e-14;
const LEAKY_RELU_SLOPE: f32 = 0.3;

/// Strictly bound MetricGAN+ VoiceBank generator.
#[derive(Debug)]
pub struct MetricGanPlus {
    weights: MetricGanWeights,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl MetricGanPlus {
    /// Loads the exact public 21-tensor generator manifest and defaults to CPU.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_metadata(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        require_metadata(file, chunks::KEY_MODEL_NAME, NAME)?;
        require_metadata(file, KEY_MODEL_CATEGORY, CATEGORY)?;
        require_metadata(file, KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF)?;
        require_metadata(file, chunks::KEY_PROVENANCE_MODEL_ID, NAME)?;

        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|value| value.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        let weights = MetricGanWeights::bind(file)?;
        Ok(Self {
            weights,
            weight_license,
            backend: BackendKind::Cpu,
        })
    }

    /// Selects the backend used by every learned projection.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Selected inference backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Stamped weight-license class, or `Unknown` when the artifact omitted it.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Enhances one complete 16 kHz mono utterance.
    ///
    /// Resampling is deliberately outside this API.  Callers must verify the
    /// input rate against [`SAMPLE_RATE`] so a plausible but wrong-rate output
    /// can never be produced silently.
    pub fn enhance(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let compute = if self.backend == BackendKind::Cpu {
            None
        } else {
            // Validate the complete learned-op set before touching PCM.  A
            // failed GPU selection cannot fall through into scalar inference.
            Some(Compute::for_backend(self.backend, METRICGAN_PLUS_HOT_OPS)?)
        };
        Ok(self.enhance_with_taps(pcm, compute.as_ref())?.waveform)
    }

    fn enhance_with_taps(&self, pcm: &[f32], compute: Option<&Compute>) -> Result<MetricGanTaps> {
        validate_pcm(pcm)?;
        let attrs = frontend_attrs();
        let spectrum = stft(pcm, &attrs)?;
        if spectrum.frames == 0 || spectrum.bins != N_BINS {
            return Err(VokraError::InvalidArgument(format!(
                "metricgan_plus: frontend emitted {} frames x {} bins; expected positive frames x {N_BINS}",
                spectrum.frames, spectrum.bins
            )));
        }

        let features = spectrum
            .re
            .iter()
            .zip(&spectrum.im)
            .map(|(&real, &imag)| (real * real + imag * imag + MAGNITUDE_EPS).sqrt().ln_1p())
            .collect::<Vec<_>>();

        let bilstm = match compute {
            Some(compute) => {
                self.weights
                    .bilstm
                    .forward_with_compute(&features, spectrum.frames, compute)?
            }
            None => self
                .weights
                .bilstm
                .forward_scalar(&features, spectrum.frames),
        };
        let mut linear1 = match compute {
            Some(compute) => {
                self.weights
                    .linear1
                    .forward_with_compute(&bilstm, spectrum.frames, compute)?
            }
            None => self
                .weights
                .linear1
                .forward_scalar(&bilstm, spectrum.frames),
        };
        for value in &mut linear1 {
            if *value < 0.0 {
                *value *= LEAKY_RELU_SLOPE;
            }
        }
        let logits = match compute {
            Some(compute) => {
                self.weights
                    .linear2
                    .forward_with_compute(&linear1, spectrum.frames, compute)?
            }
            None => self
                .weights
                .linear2
                .forward_scalar(&linear1, spectrum.frames),
        };
        let mask = logits
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                1.2 * sigmoid(self.weights.sigmoid_slope[index % N_BINS] * value)
            })
            .collect::<Vec<_>>();

        let enhanced_magnitude = features
            .iter()
            .zip(&mask)
            .map(|(&feature, &gain)| (feature * gain).exp_m1())
            .collect::<Vec<_>>();
        let mut prediction = Spectrogram {
            frames: spectrum.frames,
            bins: spectrum.bins,
            re: vec![0.0; spectrum.re.len()],
            im: vec![0.0; spectrum.im.len()],
        };
        for index in 0..enhanced_magnitude.len() {
            // SpeechBrain explicitly obtains atan2 phase, then cos/sin.  Keep
            // that contract, including atan2(0, 0) = 0 for silent bins.
            let phase = spectrum.im[index].atan2(spectrum.re[index]);
            prediction.re[index] = enhanced_magnitude[index] * phase.cos();
            prediction.im[index] = enhanced_magnitude[index] * phase.sin();
        }

        let mut inverse = IstftAttrs::new(N_FFT, HOP_LENGTH);
        inverse.win_length = N_FFT;
        inverse.window = Window::Hamming;
        inverse.window_symmetry = WindowSymmetry::Periodic;
        inverse.center = true;
        inverse.normalization = Normalization::Backward;
        inverse.real_input = true;
        inverse.length = Some(pcm.len());
        inverse.normalize_window = true;
        let mut waveform = istft(&prediction, &inverse)?;
        let peak = waveform
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        let denominator = peak + NORMALIZE_EPS;
        for value in &mut waveform {
            *value /= denominator;
        }
        if waveform.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "metricgan_plus: forward emitted a non-finite waveform".to_owned(),
            ));
        }

        Ok(MetricGanTaps {
            features,
            bilstm,
            linear1,
            mask,
            waveform,
        })
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
struct MetricGanTaps {
    features: Vec<f32>,
    bilstm: Vec<f32>,
    linear1: Vec<f32>,
    mask: Vec<f32>,
    waveform: Vec<f32>,
}

fn validate_pcm(pcm: &[f32]) -> Result<()> {
    if pcm.is_empty() {
        return Err(VokraError::InvalidArgument(
            "metricgan_plus: empty PCM input".to_owned(),
        ));
    }
    if pcm.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "metricgan_plus: PCM input contains a non-finite sample".to_owned(),
        ));
    }
    Ok(())
}

fn frontend_attrs() -> StftAttrs {
    let mut attrs = StftAttrs::new(N_FFT, HOP_LENGTH);
    attrs.win_length = N_FFT;
    attrs.window = Window::Hamming;
    attrs.window_symmetry = WindowSymmetry::Periodic;
    attrs.center = true;
    attrs.pad_mode = PadMode::Constant;
    attrs.normalization = Normalization::Backward;
    attrs.causal = false;
    attrs.real_input = true;
    attrs
}

fn require_metadata(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "metricgan_plus: missing or non-string metadata `{key}`"
            ))
        })?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "metricgan_plus: metadata `{key}` is `{actual}`, expected `{expected}`"
        )));
    }
    Ok(())
}

#[derive(Debug)]
struct MetricGanWeights {
    bilstm: BiLstmStack,
    linear1: Linear,
    linear2: Linear,
    sigmoid_slope: Vec<f32>,
}

impl MetricGanWeights {
    fn bind(file: &GgufFile) -> Result<Self> {
        let mut expected = BTreeSet::new();
        let bilstm = BiLstmStack::bind(file, &mut expected)?;
        let linear1 = Linear::bind(
            file,
            "linear1",
            2 * HIDDEN_SIZE,
            MASK_HIDDEN_SIZE,
            &mut expected,
        )?;
        let linear2 = Linear::bind(file, "linear2", MASK_HIDDEN_SIZE, N_BINS, &mut expected)?;
        let sigmoid_slope = load_tensor(file, "Learnable_sigmoid.slope", &[N_BINS], &mut expected)?;

        debug_assert_eq!(expected.len(), TENSOR_COUNT);
        let actual = file
            .tensors()
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<BTreeSet<_>>();
        let extra = actual
            .iter()
            .filter(|name| !expected.contains(**name))
            .copied()
            .collect::<Vec<_>>();
        if file.tensors().len() != TENSOR_COUNT || !extra.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "metricgan_plus: expected the exact released {TENSOR_COUNT}-tensor manifest, got {} tensors; unexpected entries: {extra:?}",
                file.tensors().len()
            )));
        }

        Ok(Self {
            bilstm,
            linear1,
            linear2,
            sigmoid_slope,
        })
    }
}

#[derive(Debug)]
struct Linear {
    input_dim: usize,
    output_dim: usize,
    weight: Vec<f32>,
    weight_transposed: Vec<f32>,
    bias: Vec<f32>,
}

impl Linear {
    fn bind(
        file: &GgufFile,
        prefix: &str,
        input_dim: usize,
        output_dim: usize,
        expected: &mut BTreeSet<String>,
    ) -> Result<Self> {
        let weight = load_tensor(
            file,
            &format!("{prefix}.weight"),
            &[output_dim, input_dim],
            expected,
        )?;
        let bias = load_tensor(file, &format!("{prefix}.bias"), &[output_dim], expected)?;
        let mut weight_transposed = vec![0.0; weight.len()];
        for output in 0..output_dim {
            for input in 0..input_dim {
                weight_transposed[input * output_dim + output] = weight[output * input_dim + input];
            }
        }
        Ok(Self {
            input_dim,
            output_dim,
            weight,
            weight_transposed,
            bias,
        })
    }

    fn forward_scalar(&self, input: &[f32], frames: usize) -> Vec<f32> {
        debug_assert_eq!(input.len(), frames * self.input_dim);
        let mut output = vec![0.0; frames * self.output_dim];
        for frame in 0..frames {
            for row in 0..self.output_dim {
                let mut sum = self.bias[row];
                let weights = &self.weight[row * self.input_dim..(row + 1) * self.input_dim];
                let values = &input[frame * self.input_dim..(frame + 1) * self.input_dim];
                for index in 0..self.input_dim {
                    sum += weights[index] * values[index];
                }
                output[frame * self.output_dim + row] = sum;
            }
        }
        output
    }

    fn forward_with_compute(
        &self,
        input: &[f32],
        frames: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if input.len() != frames * self.input_dim {
            return Err(VokraError::InvalidArgument(format!(
                "metricgan_plus linear: input has {} values, expected {} frames x {} features",
                input.len(),
                frames,
                self.input_dim
            )));
        }
        let mut output = vec![0.0; frames * self.output_dim];
        compute.gemm_f32(
            frames,
            self.output_dim,
            self.input_dim,
            input,
            &self.weight_transposed,
            Some(&self.bias),
            &mut output,
        )?;
        Ok(output)
    }
}

#[derive(Debug)]
struct BiLstmStack {
    layers: Vec<BiLstmLayer>,
}

impl BiLstmStack {
    fn bind(file: &GgufFile, expected: &mut BTreeSet<String>) -> Result<Self> {
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for layer in 0..NUM_LAYERS {
            let input_dim = if layer == 0 { N_BINS } else { 2 * HIDDEN_SIZE };
            layers.push(BiLstmLayer::bind(file, layer, input_dim, expected)?);
        }
        Ok(Self { layers })
    }

    fn forward_scalar(&self, input: &[f32], frames: usize) -> Vec<f32> {
        let mut values = input.to_vec();
        for layer in &self.layers {
            values = layer.forward_scalar(&values, frames);
        }
        values
    }

    fn forward_with_compute(
        &self,
        input: &[f32],
        frames: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        let mut values = input.to_vec();
        for layer in &self.layers {
            values = layer.forward_with_compute(&values, frames, compute)?;
        }
        Ok(values)
    }
}

#[derive(Debug)]
struct LstmDirection {
    weight_ih: Vec<f32>,
    weight_hh: Vec<f32>,
    bias_ih: Vec<f32>,
    bias_hh: Vec<f32>,
}

#[derive(Debug)]
struct BiLstmLayer {
    input_dim: usize,
    direction: [LstmDirection; 2],
}

impl BiLstmLayer {
    fn bind(
        file: &GgufFile,
        layer: usize,
        input_dim: usize,
        expected: &mut BTreeSet<String>,
    ) -> Result<Self> {
        let gates = 4 * HIDDEN_SIZE;
        let bind_direction =
            |reverse: bool, expected: &mut BTreeSet<String>| -> Result<LstmDirection> {
                let suffix = if reverse { "_reverse" } else { "" };
                Ok(LstmDirection {
                    weight_ih: load_tensor(
                        file,
                        &format!("blstm.rnn.weight_ih_l{layer}{suffix}"),
                        &[gates, input_dim],
                        expected,
                    )?,
                    weight_hh: load_tensor(
                        file,
                        &format!("blstm.rnn.weight_hh_l{layer}{suffix}"),
                        &[gates, HIDDEN_SIZE],
                        expected,
                    )?,
                    bias_ih: load_tensor(
                        file,
                        &format!("blstm.rnn.bias_ih_l{layer}{suffix}"),
                        &[gates],
                        expected,
                    )?,
                    bias_hh: load_tensor(
                        file,
                        &format!("blstm.rnn.bias_hh_l{layer}{suffix}"),
                        &[gates],
                        expected,
                    )?,
                })
            };
        Ok(Self {
            input_dim,
            direction: [
                bind_direction(false, expected)?,
                bind_direction(true, expected)?,
            ],
        })
    }

    fn forward_scalar(&self, input: &[f32], frames: usize) -> Vec<f32> {
        debug_assert_eq!(input.len(), frames * self.input_dim);
        let mut output = vec![0.0; frames * 2 * HIDDEN_SIZE];
        self.run_direction_scalar(input, frames, 0, &mut output);
        self.run_direction_scalar(input, frames, 1, &mut output);
        output
    }

    fn run_direction_scalar(
        &self,
        input: &[f32],
        frames: usize,
        direction: usize,
        output: &mut [f32],
    ) {
        let mut hidden = vec![0.0; HIDDEN_SIZE];
        let mut cell = vec![0.0; HIDDEN_SIZE];
        let mut gates = vec![0.0; 4 * HIDDEN_SIZE];
        if direction == 0 {
            for frame in 0..frames {
                let values = &input[frame * self.input_dim..(frame + 1) * self.input_dim];
                self.step_scalar(direction, values, &mut hidden, &mut cell, &mut gates);
                let base = frame * 2 * HIDDEN_SIZE;
                output[base..base + HIDDEN_SIZE].copy_from_slice(&hidden);
            }
        } else {
            for frame in (0..frames).rev() {
                let values = &input[frame * self.input_dim..(frame + 1) * self.input_dim];
                self.step_scalar(direction, values, &mut hidden, &mut cell, &mut gates);
                let base = frame * 2 * HIDDEN_SIZE + HIDDEN_SIZE;
                output[base..base + HIDDEN_SIZE].copy_from_slice(&hidden);
            }
        }
    }

    fn step_scalar(
        &self,
        direction: usize,
        input: &[f32],
        hidden: &mut [f32],
        cell: &mut [f32],
        gates: &mut [f32],
    ) {
        let weights = &self.direction[direction];
        for row in 0..4 * HIDDEN_SIZE {
            let mut sum = weights.bias_ih[row] + weights.bias_hh[row];
            let input_weights =
                &weights.weight_ih[row * self.input_dim..(row + 1) * self.input_dim];
            for index in 0..self.input_dim {
                sum += input_weights[index] * input[index];
            }
            let recurrent = &weights.weight_hh[row * HIDDEN_SIZE..(row + 1) * HIDDEN_SIZE];
            for index in 0..HIDDEN_SIZE {
                sum += recurrent[index] * hidden[index];
            }
            gates[row] = sum;
        }
        update_lstm_state(gates, hidden, cell);
    }

    fn forward_with_compute(
        &self,
        input: &[f32],
        frames: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if input.len() != frames * self.input_dim {
            return Err(VokraError::InvalidArgument(format!(
                "metricgan_plus BiLSTM: input has {} values, expected {} frames x {} features",
                input.len(),
                frames,
                self.input_dim
            )));
        }
        let mut output = vec![0.0; frames * 2 * HIDDEN_SIZE];
        self.run_direction_with_compute(input, frames, 0, &mut output, compute)?;
        self.run_direction_with_compute(input, frames, 1, &mut output, compute)?;
        Ok(output)
    }

    fn run_direction_with_compute(
        &self,
        input: &[f32],
        frames: usize,
        direction: usize,
        output: &mut [f32],
        compute: &Compute,
    ) -> Result<()> {
        let mut hidden = vec![0.0; HIDDEN_SIZE];
        let mut cell = vec![0.0; HIDDEN_SIZE];
        let mut gates = vec![0.0; 4 * HIDDEN_SIZE];
        let mut recurrent = vec![0.0; 4 * HIDDEN_SIZE];
        if direction == 0 {
            for frame in 0..frames {
                let values = &input[frame * self.input_dim..(frame + 1) * self.input_dim];
                self.step_with_compute(
                    direction,
                    values,
                    &mut hidden,
                    &mut cell,
                    &mut gates,
                    &mut recurrent,
                    compute,
                )?;
                let base = frame * 2 * HIDDEN_SIZE;
                output[base..base + HIDDEN_SIZE].copy_from_slice(&hidden);
            }
        } else {
            for frame in (0..frames).rev() {
                let values = &input[frame * self.input_dim..(frame + 1) * self.input_dim];
                self.step_with_compute(
                    direction,
                    values,
                    &mut hidden,
                    &mut cell,
                    &mut gates,
                    &mut recurrent,
                    compute,
                )?;
                let base = frame * 2 * HIDDEN_SIZE + HIDDEN_SIZE;
                output[base..base + HIDDEN_SIZE].copy_from_slice(&hidden);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn step_with_compute(
        &self,
        direction: usize,
        input: &[f32],
        hidden: &mut [f32],
        cell: &mut [f32],
        gates: &mut [f32],
        recurrent: &mut [f32],
        compute: &Compute,
    ) -> Result<()> {
        let weights = &self.direction[direction];
        compute.gemv_f32(
            4 * HIDDEN_SIZE,
            self.input_dim,
            &weights.weight_ih,
            input,
            Some(&weights.bias_ih),
            gates,
        )?;
        compute.gemv_f32(
            4 * HIDDEN_SIZE,
            HIDDEN_SIZE,
            &weights.weight_hh,
            hidden,
            Some(&weights.bias_hh),
            recurrent,
        )?;
        for (gate, &value) in gates.iter_mut().zip(recurrent.iter()) {
            *gate += value;
        }
        update_lstm_state(gates, hidden, cell);
        Ok(())
    }
}

fn update_lstm_state(gates: &[f32], hidden: &mut [f32], cell: &mut [f32]) {
    for index in 0..HIDDEN_SIZE {
        let input_gate = sigmoid(gates[index]);
        let forget_gate = sigmoid(gates[HIDDEN_SIZE + index]);
        let candidate = gates[2 * HIDDEN_SIZE + index].tanh();
        let output_gate = sigmoid(gates[3 * HIDDEN_SIZE + index]);
        let next_cell = forget_gate * cell[index] + input_gate * candidate;
        cell[index] = next_cell;
        hidden[index] = output_gate * next_cell.tanh();
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn load_tensor(
    file: &GgufFile,
    name: &str,
    expected_shape: &[usize],
    expected_names: &mut BTreeSet<String>,
) -> Result<Vec<f32>> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "metricgan_plus: required tensor `{name}` is missing; expected shape {expected_shape:?}"
        ))
    })?;
    let actual_shape = info
        .dimensions
        .iter()
        .map(|&dimension| dimension as usize)
        .collect::<Vec<_>>();
    if actual_shape != expected_shape {
        return Err(VokraError::ModelLoad(format!(
            "metricgan_plus: tensor `{name}` has shape {actual_shape:?}, expected {expected_shape:?}"
        )));
    }
    if !matches!(info.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        return Err(VokraError::ModelLoad(format!(
            "metricgan_plus: tensor `{name}` uses {:?}; only the converter's lossless F32/F16/BF16 pass-through types are accepted until quantized real-weight parity exists",
            info.dtype
        )));
    }
    let values = file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!(
            "metricgan_plus: tensor `{name}` decode failed: {error}"
        ))
    })?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(format!(
            "metricgan_plus: tensor `{name}` contains a non-finite value"
        )));
    }
    expected_names.insert(name.to_owned());
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    fn f32_bytes(len: usize, scale: f32) -> Vec<u8> {
        (0..len)
            .flat_map(|index| {
                let centered = (index % 17) as f32 - 8.0;
                (centered * scale).to_le_bytes()
            })
            .collect()
    }

    fn add_tensor(builder: &mut GgufBuilder, name: &str, shape: &[usize], scale: f32) {
        let len = shape.iter().product();
        builder
            .add_tensor(
                name,
                GgmlType::F32,
                shape.iter().map(|&dimension| dimension as u64).collect(),
                f32_bytes(len, scale),
            )
            .unwrap();
    }

    fn fixture(extra: bool, wrong_slope: bool) -> GgufFile {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        builder.add_string(chunks::KEY_MODEL_NAME, NAME);
        builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        builder.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
        builder.add_string(chunks::KEY_PROVENANCE_MODEL_ID, NAME);
        builder.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        );
        for layer in 0..NUM_LAYERS {
            let input_dim = if layer == 0 { N_BINS } else { 2 * HIDDEN_SIZE };
            for suffix in ["", "_reverse"] {
                add_tensor(
                    &mut builder,
                    &format!("blstm.rnn.weight_ih_l{layer}{suffix}"),
                    &[4 * HIDDEN_SIZE, input_dim],
                    1.0e-5,
                );
                add_tensor(
                    &mut builder,
                    &format!("blstm.rnn.weight_hh_l{layer}{suffix}"),
                    &[4 * HIDDEN_SIZE, HIDDEN_SIZE],
                    1.0e-5,
                );
                add_tensor(
                    &mut builder,
                    &format!("blstm.rnn.bias_ih_l{layer}{suffix}"),
                    &[4 * HIDDEN_SIZE],
                    1.0e-5,
                );
                add_tensor(
                    &mut builder,
                    &format!("blstm.rnn.bias_hh_l{layer}{suffix}"),
                    &[4 * HIDDEN_SIZE],
                    1.0e-5,
                );
            }
        }
        add_tensor(
            &mut builder,
            "linear1.weight",
            &[MASK_HIDDEN_SIZE, 2 * HIDDEN_SIZE],
            1.0e-5,
        );
        add_tensor(&mut builder, "linear1.bias", &[MASK_HIDDEN_SIZE], 1.0e-5);
        add_tensor(
            &mut builder,
            "linear2.weight",
            &[N_BINS, MASK_HIDDEN_SIZE],
            1.0e-5,
        );
        add_tensor(&mut builder, "linear2.bias", &[N_BINS], 1.0e-5);
        add_tensor(
            &mut builder,
            "Learnable_sigmoid.slope",
            if wrong_slope {
                &[N_BINS - 1]
            } else {
                &[N_BINS]
            },
            0.0,
        );
        if extra {
            add_tensor(&mut builder, "discriminator.weight", &[1], 0.0);
        }
        GgufFile::parse(builder.to_bytes().unwrap()).unwrap()
    }

    fn input_pcm() -> Vec<f32> {
        (0..1024)
            .map(|index| {
                let phase = index as f32 * 2.0 * std::f32::consts::PI * 440.0 / SAMPLE_RATE as f32;
                0.25 * phase.sin()
            })
            .collect()
    }

    fn read_f32_file(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|error| panic!("read MetricGAN+ fixture {}: {error}", path.display()));
        assert_eq!(
            bytes.len() % 4,
            0,
            "MetricGAN+ fixture {} is not raw f32",
            path.display()
        );
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn compare_official(label: &str, actual: &[f32], expected: &[f32]) {
        const MAX_ABS_BOUND: f32 = 0.01;
        const MEAN_ABS_BOUND: f32 = 0.001;

        assert_eq!(actual.len(), expected.len(), "{label} length");
        assert!(
            actual.iter().all(|value| value.is_finite()),
            "{label} finite"
        );
        let (index, max_abs) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .map(|(index, (&actual, &expected))| (index, (actual - expected).abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap();
        let mean_abs = actual
            .iter()
            .zip(expected)
            .map(|(&actual, &expected)| (actual - expected).abs())
            .sum::<f32>()
            / actual.len() as f32;
        eprintln!(
            "MetricGAN+ {label}: max_abs={max_abs:.9e} at {index} \
             (actual={:.9e}, reference={:.9e}), mean_abs={mean_abs:.9e}",
            actual[index], expected[index]
        );
        assert!(
            max_abs <= MAX_ABS_BOUND,
            "{label} max_abs={max_abs:.9e}, bound={MAX_ABS_BOUND:.9e}"
        );
        assert!(
            mean_abs <= MEAN_ABS_BOUND,
            "{label} mean_abs={mean_abs:.9e}, bound={MEAN_ABS_BOUND:.9e}"
        );
    }

    #[test]
    fn strict_manifest_binds_and_cpu_forward_is_finite() {
        let model = MetricGanPlus::from_gguf(&fixture(false, false)).unwrap();
        assert_eq!(model.weight_license(), LicenseClass::Permissive);
        let output = model.enhance(&input_pcm()).unwrap();
        assert_eq!(output.len(), 1024);
        assert!(output.iter().all(|value| value.is_finite()));
        let peak = output.iter().map(|value| value.abs()).fold(0.0, f32::max);
        assert!((peak - 1.0).abs() < 1.0e-5, "peak={peak}");
    }

    #[test]
    fn wrong_shape_and_extra_tensor_fail_loudly() {
        let wrong = MetricGanPlus::from_gguf(&fixture(false, true)).unwrap_err();
        assert!(
            wrong.to_string().contains("Learnable_sigmoid.slope")
                && wrong.to_string().contains("shape")
        );
        let extra = MetricGanPlus::from_gguf(&fixture(true, false)).unwrap_err();
        assert!(
            extra
                .to_string()
                .contains("exact released 21-tensor manifest")
                && extra.to_string().contains("discriminator.weight")
        );
    }

    #[test]
    fn empty_and_non_finite_pcm_fail_loudly() {
        let model = MetricGanPlus::from_gguf(&fixture(false, false)).unwrap();
        assert!(
            model
                .enhance(&[])
                .unwrap_err()
                .to_string()
                .contains("empty")
        );
        assert!(
            model
                .enhance(&[f32::NAN])
                .unwrap_err()
                .to_string()
                .contains("non-finite")
        );
    }

    #[test]
    fn unavailable_backend_never_falls_back_to_cpu() {
        let model = MetricGanPlus::from_gguf(&fixture(false, false))
            .unwrap()
            .with_backend(BackendKind::Vulkan);
        let error = model.enhance(&input_pcm()).unwrap_err().to_string();
        assert!(
            error.contains("vulkan") || error.contains("Vulkan"),
            "unexpected backend error: {error}"
        );
    }

    #[test]
    fn real_checkpoint_matches_official_speechbrain_when_requested() {
        let gguf = std::env::var_os("VOKRA_METRICGAN_PLUS_GGUF");
        let reference = std::env::var_os("VOKRA_METRICGAN_PLUS_REFERENCE_DIR");
        let (gguf, reference) = match (gguf, reference) {
            (Some(gguf), Some(reference)) => (gguf, reference),
            (None, None) => {
                eprintln!(
                    "skipping real MetricGAN+ parity: set VOKRA_METRICGAN_PLUS_GGUF and \
                     VOKRA_METRICGAN_PLUS_REFERENCE_DIR"
                );
                return;
            }
            _ => {
                panic!(
                    "set both VOKRA_METRICGAN_PLUS_GGUF and \
                     VOKRA_METRICGAN_PLUS_REFERENCE_DIR; a half-enabled real parity run is invalid"
                );
            }
        };
        let reference = std::path::PathBuf::from(reference);
        let file = GgufFile::open(gguf).expect("open public MetricGAN+ GGUF");
        let model = MetricGanPlus::from_gguf(&file).expect("strict real MetricGAN+ bind");
        let pcm = read_f32_file(&reference.join("pcm.f32.bin"));
        assert_eq!(pcm.len(), 4_096);
        let taps = model
            .enhance_with_taps(&pcm, None)
            .expect("CPU real forward");

        for (label, actual) in [
            ("features", taps.features.as_slice()),
            ("bilstm", taps.bilstm.as_slice()),
            ("linear1", taps.linear1.as_slice()),
            ("mask", taps.mask.as_slice()),
            ("waveform", taps.waveform.as_slice()),
        ] {
            let expected = read_f32_file(&reference.join(format!("{label}.f32.bin")));
            compare_official(label, actual, &expected);
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metricgan_plus_metal_matches_cpu_when_device_is_available() {
        if vokra_backend_metal::vokra_metal_probe().is_err() {
            eprintln!("skipping MetricGAN+ Metal parity: no system Metal device");
            return;
        }
        let pcm = input_pcm();
        let cpu_model = MetricGanPlus::from_gguf(&fixture(false, false)).unwrap();
        let cpu = cpu_model.enhance_with_taps(&pcm, None).unwrap();
        let metal_model = MetricGanPlus::from_gguf(&fixture(false, false)).unwrap();
        let compute = Compute::for_backend(BackendKind::Metal, METRICGAN_PLUS_HOT_OPS).unwrap();
        let metal = metal_model.enhance_with_taps(&pcm, Some(&compute)).unwrap();

        // Pre-registered before the first device run.  The repository's
        // general FP32 backend boundary is 0.01; this compact graph is held to
        // a ten-times tighter whole-model limit.
        for (name, lhs, rhs) in [
            ("bilstm", &cpu.bilstm, &metal.bilstm),
            ("linear1", &cpu.linear1, &metal.linear1),
            ("mask", &cpu.mask, &metal.mask),
            ("waveform", &cpu.waveform, &metal.waveform),
        ] {
            let max = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max <= 1.0e-3, "MetricGAN+ {name} CPU/Metal max={max}");
        }
        assert_eq!(cpu.features, metal.features);
    }
}
