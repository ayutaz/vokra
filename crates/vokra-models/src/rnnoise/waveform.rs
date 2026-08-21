//! Xiph RNNoise v0.2 waveform analysis and synthesis.
//!
//! This is a direct native transcription of the float build in upstream
//! `src/denoise.c`, `src/pitch.c`, and `src/celt_lpc.c` at tag `v0.2`.
//! The trained frontend is model-specific: 32 ERB bands, 65 features, a
//! 1,728-sample pitch buffer, and one-frame delayed spectrum processing.

use vokra_core::engines::{DenoiseEngine, DenoiseStreamHandle};
use vokra_core::{Result, VokraError};
use vokra_ops::fft::{Complex32, RealFftPlan};

use super::{
    FRAME_SIZE, N_BANDS, N_FEATURES, RnnoiseNetworkState, RnnoiseV02, SAMPLE_RATE, WINDOW_SIZE,
};

const FREQ_SIZE: usize = WINDOW_SIZE / 2 + 1;
const PITCH_MIN_PERIOD: usize = 60;
const PITCH_MAX_PERIOD: usize = 768;
const PITCH_FRAME_SIZE: usize = 960;
const PITCH_BUF_SIZE: usize = PITCH_MAX_PERIOD + PITCH_FRAME_SIZE;

/// v0.2 ERB-band edges in 50 Hz FFT bins.  The two guard bands are part of
/// upstream's triangular energy distribution and must not be dropped.
const BAND_EDGES: [usize; N_BANDS + 2] = [
    0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 28, 32, 36, 41, 47, 53, 60, 68, 77, 87, 98, 110, 124,
    140, 157, 176, 198, 223, 251, 282, 317, 356, 400,
];

/// Result of one exact 480-sample upstream frame step.
#[derive(Debug, Clone, PartialEq)]
pub struct RnnoiseFrameOutput {
    /// Enhanced PCM for the delayed spectrum.  The first frame is zero due to
    /// the algorithm's intentional one-frame lookahead.
    pub pcm: [f32; FRAME_SIZE],
    /// Voice probability emitted for the current analysis frame.
    pub vad_probability: f32,
}

/// Stateful RNNoise v0.2 waveform stream.
pub struct RnnoiseStream {
    model: RnnoiseV02,
    network: RnnoiseNetworkState,
    fft: RealFftPlan,
    half_window: [f32; FRAME_SIZE],
    analysis_mem: [f32; FRAME_SIZE],
    synthesis_mem: [f32; FRAME_SIZE],
    pitch_buf: [f32; PITCH_BUF_SIZE],
    last_gain: f32,
    last_period: usize,
    highpass_mem: [f32; 2],
    last_gains: [f32; N_BANDS],
    delayed_x: [Complex32; FREQ_SIZE],
    delayed_p: [Complex32; FREQ_SIZE],
    delayed_ex: [f32; N_BANDS],
    delayed_ep: [f32; N_BANDS],
    delayed_exp: [f32; N_BANDS],
    pending: Vec<f32>,
}

impl RnnoiseStream {
    pub(super) fn new(model: RnnoiseV02) -> Self {
        Self {
            model,
            network: RnnoiseNetworkState::default(),
            fft: RealFftPlan::new(WINDOW_SIZE),
            half_window: std::array::from_fn(half_window_value),
            analysis_mem: [0.0; FRAME_SIZE],
            synthesis_mem: [0.0; FRAME_SIZE],
            pitch_buf: [0.0; PITCH_BUF_SIZE],
            last_gain: 0.0,
            last_period: 0,
            highpass_mem: [0.0; 2],
            last_gains: [0.0; N_BANDS],
            delayed_x: [Complex32::ZERO; FREQ_SIZE],
            delayed_p: [Complex32::ZERO; FREQ_SIZE],
            delayed_ex: [0.0; N_BANDS],
            delayed_ep: [0.0; N_BANDS],
            delayed_exp: [0.0; N_BANDS],
            pending: Vec::new(),
        }
    }

    /// Processes exactly one official 10 ms frame.
    pub fn process_frame(&mut self, input: &[f32; FRAME_SIZE]) -> Result<RnnoiseFrameOutput> {
        if input.iter().any(|sample| !sample.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "rnnoise: PCM frame contains a non-finite sample".to_owned(),
            ));
        }

        let filtered = self.highpass(input);
        let (x, p, ex, ep, exp, features, silence) = self.compute_features(&filtered);
        let mut vad_probability = 0.0;

        if !silence {
            let decision = self.model.forward_features(&mut self.network, &features)?;
            vad_probability = decision.vad_probability;
            pitch_filter(
                &mut self.delayed_x,
                &self.delayed_p,
                &self.delayed_ex,
                &self.delayed_ep,
                &self.delayed_exp,
                &decision.gains,
            );
            let mut gains = decision.gains;
            for (gain, previous) in gains.iter_mut().zip(&mut self.last_gains) {
                *gain = gain.max(0.6 * *previous);
                *previous = *gain;
            }
            let bins = interp_band_gain(&gains);
            for (value, gain) in self.delayed_x.iter_mut().zip(bins) {
                *value = value.scale(gain);
            }
        }

        let pcm = self.synthesize_delayed();
        self.delayed_x = x;
        self.delayed_p = p;
        self.delayed_ex = ex;
        self.delayed_ep = ep;
        self.delayed_exp = exp;
        Ok(RnnoiseFrameOutput {
            pcm,
            vad_probability,
        })
    }

    pub(super) fn pending_is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub(super) fn flush_partial(&mut self) -> Result<Vec<f32>> {
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        let pending_len = self.pending.len();
        let mut frame = [0.0; FRAME_SIZE];
        frame[..pending_len].copy_from_slice(&self.pending);
        self.pending.clear();
        let result = self.process_frame(&frame)?;
        Ok(result.pcm[..pending_len].to_vec())
    }

    fn compute_features(
        &mut self,
        input: &[f32; FRAME_SIZE],
    ) -> (
        [Complex32; FREQ_SIZE],
        [Complex32; FREQ_SIZE],
        [f32; N_BANDS],
        [f32; N_BANDS],
        [f32; N_BANDS],
        [f32; N_FEATURES],
        bool,
    ) {
        let x = self.frame_analysis(input);
        let ex = compute_band_energy(&x);

        self.pitch_buf.copy_within(FRAME_SIZE.., 0);
        self.pitch_buf[PITCH_BUF_SIZE - FRAME_SIZE..].copy_from_slice(input);
        let pitch_lp = pitch_downsample(&self.pitch_buf);
        let searched = pitch_search(
            &pitch_lp[PITCH_MAX_PERIOD / 2..],
            &pitch_lp,
            PITCH_FRAME_SIZE,
            PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD,
        );
        let mut pitch_index = PITCH_MAX_PERIOD - searched;
        let gain = remove_doubling(
            &pitch_lp,
            &mut pitch_index,
            self.last_period,
            self.last_gain,
        );
        self.last_period = pitch_index;
        self.last_gain = gain;

        let pitch_start = PITCH_BUF_SIZE - WINDOW_SIZE - pitch_index;
        let mut pitch_frame = [0.0; WINDOW_SIZE];
        pitch_frame.copy_from_slice(&self.pitch_buf[pitch_start..pitch_start + WINDOW_SIZE]);
        apply_window(&mut pitch_frame, &self.half_window);
        let p = forward_transform(&self.fft, &pitch_frame);
        let ep = compute_band_energy(&p);
        let mut exp = compute_band_corr(&x, &p);
        for band in 0..N_BANDS {
            exp[band] /= (0.001 + ex[band] * ep[band]).sqrt();
        }

        let mut features = [0.0; N_FEATURES];
        let pitch_dct = dct(&exp);
        features[N_BANDS..2 * N_BANDS].copy_from_slice(&pitch_dct);
        features[2 * N_BANDS] = 0.01 * (pitch_index as f32 - 300.0);

        let mut energy = 0.0f32;
        let mut log_energy = [0.0; N_BANDS];
        let mut log_max = -2.0f32;
        let mut follow = -2.0f32;
        for band in 0..N_BANDS {
            let raw = (0.01 + ex[band]).log10();
            log_energy[band] = raw.max(log_max - 7.0).max(follow - 1.5);
            log_max = log_max.max(log_energy[band]);
            follow = (follow - 1.5).max(log_energy[band]);
            energy += ex[band];
        }
        let silence = energy < 0.04;
        if !silence {
            let energy_dct = dct(&log_energy);
            features[..N_BANDS].copy_from_slice(&energy_dct);
            features[0] -= 12.0;
            features[1] -= 4.0;
        }
        (x, p, ex, ep, exp, features, silence)
    }

    fn frame_analysis(&mut self, input: &[f32; FRAME_SIZE]) -> [Complex32; FREQ_SIZE] {
        let mut frame = [0.0; WINDOW_SIZE];
        frame[..FRAME_SIZE].copy_from_slice(&self.analysis_mem);
        frame[FRAME_SIZE..].copy_from_slice(input);
        self.analysis_mem.copy_from_slice(input);
        apply_window(&mut frame, &self.half_window);
        forward_transform(&self.fft, &frame)
    }

    fn synthesize_delayed(&mut self) -> [f32; FRAME_SIZE] {
        let mut frame = inverse_transform(&self.fft, &self.delayed_x);
        apply_window(&mut frame, &self.half_window);
        let mut output = [0.0; FRAME_SIZE];
        for index in 0..FRAME_SIZE {
            output[index] = frame[index] + self.synthesis_mem[index];
        }
        self.synthesis_mem.copy_from_slice(&frame[FRAME_SIZE..]);
        output
    }

    fn highpass(&mut self, input: &[f32; FRAME_SIZE]) -> [f32; FRAME_SIZE] {
        let mut output = [0.0; FRAME_SIZE];
        for (out, &sample) in output.iter_mut().zip(input) {
            let value = sample + self.highpass_mem[0];
            self.highpass_mem[0] = (f64::from(self.highpass_mem[1]) - 2.0 * f64::from(sample)
                + 1.99599 * f64::from(value)) as f32;
            self.highpass_mem[1] = (f64::from(sample) - 0.99600 * f64::from(value)) as f32;
            *out = value;
        }
        output
    }
}

impl DenoiseEngine for RnnoiseV02 {
    fn open_stream(&self, sample_rate: u32) -> Result<Box<dyn DenoiseStreamHandle + Send>> {
        if sample_rate != SAMPLE_RATE {
            return Err(VokraError::InvalidArgument(format!(
                "rnnoise: sample rate mismatch — requested {sample_rate} Hz but v0.2 is fixed at {SAMPLE_RATE} Hz (resample upstream)"
            )));
        }
        Ok(Box::new(self.stream()))
    }
}

impl DenoiseStreamHandle for RnnoiseStream {
    fn push_pcm(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        if pcm.iter().any(|sample| !sample.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "rnnoise: PCM input contains a non-finite sample".to_owned(),
            ));
        }
        self.pending.extend_from_slice(pcm);
        let frames = self.pending.len() / FRAME_SIZE;
        let mut output = Vec::with_capacity(frames * FRAME_SIZE);
        for frame_index in 0..frames {
            let start = frame_index * FRAME_SIZE;
            let frame: [f32; FRAME_SIZE] = self.pending[start..start + FRAME_SIZE]
                .try_into()
                .expect("exact RNNoise frame");
            output.extend_from_slice(&self.process_frame(&frame)?.pcm);
        }
        if frames != 0 {
            self.pending.drain(..frames * FRAME_SIZE);
        }
        Ok(output)
    }

    fn reset(&mut self) {
        let model = self.model.clone();
        *self = Self::new(model);
    }
}

fn half_window_value(index: usize) -> f32 {
    let phase = 0.5 * std::f64::consts::PI * (index as f64 + 0.5) / FRAME_SIZE as f64;
    (0.5 * std::f64::consts::PI * phase.sin().powi(2)).sin() as f32
}

fn apply_window(frame: &mut [f32; WINDOW_SIZE], half: &[f32; FRAME_SIZE]) {
    for index in 0..FRAME_SIZE {
        frame[index] *= half[index];
        frame[WINDOW_SIZE - 1 - index] *= half[index];
    }
}

fn forward_transform(fft: &RealFftPlan, input: &[f32; WINDOW_SIZE]) -> [Complex32; FREQ_SIZE] {
    let scale = 1.0 / WINDOW_SIZE as f32;
    fft.forward(input)
        .into_iter()
        .map(|value| value.scale(scale))
        .collect::<Vec<_>>()
        .try_into()
        .expect("960-point real FFT has 481 bins")
}

fn inverse_transform(fft: &RealFftPlan, spectrum: &[Complex32; FREQ_SIZE]) -> [f32; WINDOW_SIZE] {
    fft.inverse(spectrum)
        .into_iter()
        .map(|value| value * WINDOW_SIZE as f32)
        .collect::<Vec<_>>()
        .try_into()
        .expect("960-point inverse FFT has 960 samples")
}

fn compute_band_energy(spectrum: &[Complex32; FREQ_SIZE]) -> [f32; N_BANDS] {
    let mut sums = [0.0f32; N_BANDS + 2];
    for band in 0..N_BANDS + 1 {
        let width = BAND_EDGES[band + 1] - BAND_EDGES[band];
        for offset in 0..width {
            let fraction = offset as f32 / width as f32;
            let value = spectrum[BAND_EDGES[band] + offset];
            let power = value.re * value.re + value.im * value.im;
            sums[band] += (1.0 - fraction) * power;
            sums[band + 1] += fraction * power;
        }
    }
    sums[1] = (sums[0] + sums[1]) * (2.0 / 3.0);
    sums[N_BANDS] = (sums[N_BANDS] + sums[N_BANDS + 1]) * (2.0 / 3.0);
    std::array::from_fn(|band| sums[band + 1])
}

fn compute_band_corr(
    left: &[Complex32; FREQ_SIZE],
    right: &[Complex32; FREQ_SIZE],
) -> [f32; N_BANDS] {
    let mut sums = [0.0f32; N_BANDS + 2];
    for band in 0..N_BANDS + 1 {
        let width = BAND_EDGES[band + 1] - BAND_EDGES[band];
        for offset in 0..width {
            let fraction = offset as f32 / width as f32;
            let index = BAND_EDGES[band] + offset;
            let value = left[index].re * right[index].re + left[index].im * right[index].im;
            sums[band] += (1.0 - fraction) * value;
            sums[band + 1] += fraction * value;
        }
    }
    sums[1] = (sums[0] + sums[1]) * (2.0 / 3.0);
    sums[N_BANDS] = (sums[N_BANDS] + sums[N_BANDS + 1]) * (2.0 / 3.0);
    std::array::from_fn(|band| sums[band + 1])
}

fn interp_band_gain(bands: &[f32; N_BANDS]) -> [f32; FREQ_SIZE] {
    let mut bins = [0.0; FREQ_SIZE];
    for band in 1..N_BANDS {
        let width = BAND_EDGES[band + 1] - BAND_EDGES[band];
        for offset in 0..width {
            let fraction = offset as f32 / width as f32;
            bins[BAND_EDGES[band] + offset] =
                (1.0 - fraction) * bands[band - 1] + fraction * bands[band];
        }
    }
    bins[..BAND_EDGES[1]].fill(bands[0]);
    bins[BAND_EDGES[N_BANDS]..BAND_EDGES[N_BANDS + 1]].fill(bands[N_BANDS - 1]);
    bins
}

fn dct(input: &[f32; N_BANDS]) -> [f32; N_BANDS] {
    let scale = (2.0f64 / 22.0).sqrt();
    std::array::from_fn(|column| {
        let mut sum = 0.0f32;
        for (row, &value) in input.iter().enumerate() {
            let mut table =
                ((row as f64 + 0.5) * column as f64 * std::f64::consts::PI / N_BANDS as f64).cos();
            if column == 0 {
                table *= 0.5f64.sqrt();
            }
            sum += value * table as f32;
        }
        (f64::from(sum) * scale) as f32
    })
}

fn pitch_downsample(input: &[f32; PITCH_BUF_SIZE]) -> [f32; PITCH_BUF_SIZE / 2] {
    let mut output = [0.0; PITCH_BUF_SIZE / 2];
    output[0] = 0.5 * (0.5 * input[1] + input[0]);
    for index in 1..PITCH_BUF_SIZE / 2 {
        output[index] =
            0.5 * (0.5 * (input[2 * index - 1] + input[2 * index + 1]) + input[2 * index]);
    }

    let mut autocorr = [0.0; 5];
    for lag in 0..=4 {
        for index in lag..output.len() {
            autocorr[lag] += output[index] * output[index - lag];
        }
    }
    autocorr[0] *= 1.0001;
    for (lag, value) in autocorr.iter_mut().enumerate().skip(1) {
        *value -= *value * (0.008 * lag as f32).powi(2);
    }
    let mut lpc = lpc(&autocorr);
    let mut decay = 1.0f32;
    for value in &mut lpc {
        decay *= 0.9;
        *value *= decay;
    }
    let coefficients = [
        lpc[0] + 0.8,
        lpc[1] + 0.8 * lpc[0],
        lpc[2] + 0.8 * lpc[1],
        lpc[3] + 0.8 * lpc[2],
        0.8 * lpc[3],
    ];
    let mut memory = [0.0; 5];
    for sample in &mut output {
        let input_sample = *sample;
        let sum = input_sample
            + coefficients[0] * memory[0]
            + coefficients[1] * memory[1]
            + coefficients[2] * memory[2]
            + coefficients[3] * memory[3]
            + coefficients[4] * memory[4];
        memory.copy_within(0..4, 1);
        memory[0] = input_sample;
        *sample = sum;
    }
    output
}

fn lpc(autocorr: &[f32; 5]) -> [f32; 4] {
    let mut output = [0.0; 4];
    let mut error = autocorr[0];
    if autocorr[0] == 0.0 {
        return output;
    }
    for index in 0..4 {
        let mut rr = 0.0f32;
        for inner in 0..index {
            rr += output[inner] * autocorr[index - inner];
        }
        rr += autocorr[index + 1];
        let reflection = -rr / error;
        output[index] = reflection;
        for inner in 0..(index + 1) / 2 {
            let left = output[inner];
            let right = output[index - 1 - inner];
            output[inner] = left + reflection * right;
            output[index - 1 - inner] = right + reflection * left;
        }
        error -= reflection * reflection * error;
        if error < 0.001 * autocorr[0] {
            break;
        }
    }
    output
}

fn inner_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .fold(0.0, |sum, (&a, &b)| sum + a * b)
}

fn pitch_xcorr(x: &[f32], y: &[f32], len: usize, max_pitch: usize) -> Vec<f32> {
    (0..max_pitch)
        .map(|lag| inner_product(&x[..len], &y[lag..lag + len]))
        .collect()
}

fn find_best_pitch(correlation: &[f32], y: &[f32], len: usize, max_pitch: usize) -> [usize; 2] {
    let mut energy = 1.0f32 + y[..len].iter().map(|value| value * value).sum::<f32>();
    let mut best_num = [-1.0f32; 2];
    let mut best_den = [0.0f32; 2];
    let mut best = [0usize, 1usize];
    for lag in 0..max_pitch {
        if correlation[lag] > 0.0 {
            let scaled = correlation[lag] * 1e-12;
            let numerator = scaled * scaled;
            if numerator * best_den[1] > best_num[1] * energy {
                if numerator * best_den[0] > best_num[0] * energy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best[1] = best[0];
                    best_num[0] = numerator;
                    best_den[0] = energy;
                    best[0] = lag;
                } else {
                    best_num[1] = numerator;
                    best_den[1] = energy;
                    best[1] = lag;
                }
            }
        }
        energy += y[lag + len] * y[lag + len] - y[lag] * y[lag];
        energy = energy.max(1.0);
    }
    best
}

fn pitch_search(x: &[f32], y: &[f32], len: usize, max_pitch: usize) -> usize {
    let lag = len + max_pitch;
    let x_quarter: Vec<f32> = (0..len / 4).map(|index| x[2 * index]).collect();
    let y_quarter: Vec<f32> = (0..lag / 4).map(|index| y[2 * index]).collect();
    let coarse = pitch_xcorr(&x_quarter, &y_quarter, len / 4, max_pitch / 4);
    let coarse_best = find_best_pitch(&coarse, &y_quarter, len / 4, max_pitch / 4);

    let mut fine = vec![0.0f32; max_pitch / 2];
    for (lag, value) in fine.iter_mut().enumerate() {
        if lag.abs_diff(2 * coarse_best[0]) <= 2 || lag.abs_diff(2 * coarse_best[1]) <= 2 {
            *value = inner_product(&x[..len / 2], &y[lag..lag + len / 2]).max(-1.0);
        }
    }
    let best = find_best_pitch(&fine, y, len / 2, max_pitch / 2)[0];
    let offset = if best > 0 && best < max_pitch / 2 - 1 {
        let a = fine[best - 1];
        let b = fine[best];
        let c = fine[best + 1];
        if c - a > 0.7 * (b - a) {
            1isize
        } else if a - c > 0.7 * (b - c) {
            -1
        } else {
            0
        }
    } else {
        0
    };
    (2 * best).saturating_add_signed(-offset)
}

fn pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    xy / (1.0 + xx * yy).sqrt()
}

fn remove_doubling(
    x: &[f32; PITCH_BUF_SIZE / 2],
    period: &mut usize,
    previous_period: usize,
    previous_gain: f32,
) -> f32 {
    const SECOND_CHECK: [usize; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];
    let min_period_original = PITCH_MIN_PERIOD;
    let max_period = PITCH_MAX_PERIOD / 2;
    let min_period = PITCH_MIN_PERIOD / 2;
    let mut candidate = *period / 2;
    let previous_period = previous_period / 2;
    let length = PITCH_FRAME_SIZE / 2;
    if candidate >= max_period {
        candidate = max_period - 1;
    }
    let origin = max_period;
    let current = &x[origin..origin + length];
    let xx = inner_product(current, current);
    let mut yy_lookup = [0.0f32; PITCH_MAX_PERIOD + 1];
    yy_lookup[0] = xx;
    let mut yy = xx;
    for lag in 1..=max_period {
        yy +=
            x[origin - lag] * x[origin - lag] - x[origin + length - lag] * x[origin + length - lag];
        yy_lookup[lag] = yy.max(0.0);
    }
    let xy = inner_product(current, &x[origin - candidate..origin - candidate + length]);
    let mut best_xy = xy;
    let mut best_yy = yy_lookup[candidate];
    let base_gain = pitch_gain(xy, xx, best_yy);
    let mut gain = base_gain;
    let mut best_period = candidate;

    for divisor in 2..=15 {
        let first = (2 * candidate + divisor) / (2 * divisor);
        if first < min_period {
            break;
        }
        let second = if divisor == 2 {
            if first + candidate > max_period {
                candidate
            } else {
                candidate + first
            }
        } else {
            (2 * SECOND_CHECK[divisor] * candidate + divisor) / (2 * divisor)
        };
        let xy_first = inner_product(current, &x[origin - first..origin - first + length]);
        let xy_second = inner_product(current, &x[origin - second..origin - second + length]);
        let test_xy = 0.5 * (xy_first + xy_second);
        let test_yy = 0.5 * (yy_lookup[first] + yy_lookup[second]);
        let test_gain = pitch_gain(test_xy, xx, test_yy);
        let continuation = if first.abs_diff(previous_period) <= 1 {
            previous_gain
        } else if first.abs_diff(previous_period) <= 2 && 5 * divisor * divisor < candidate {
            0.5 * previous_gain
        } else {
            0.0
        };
        let mut threshold = 0.3f32.max(0.7 * base_gain - continuation);
        if first < 3 * min_period {
            threshold = 0.4f32.max(0.85 * base_gain - continuation);
        } else if first < 2 * min_period {
            threshold = 0.5f32.max(0.9 * base_gain - continuation);
        }
        if test_gain > threshold {
            best_xy = test_xy;
            best_yy = test_yy;
            best_period = first;
            gain = test_gain;
        }
    }

    best_xy = best_xy.max(0.0);
    let mut pitch_gain = if best_yy <= best_xy {
        1.0
    } else {
        best_xy / (best_yy + 1.0)
    };
    let correlations: [f32; 3] = std::array::from_fn(|offset| {
        let lag = best_period + offset - 1;
        inner_product(current, &x[origin - lag..origin - lag + length])
    });
    let offset = if correlations[2] - correlations[0] > 0.7 * (correlations[1] - correlations[0]) {
        1isize
    } else if correlations[0] - correlations[2] > 0.7 * (correlations[1] - correlations[2]) {
        -1
    } else {
        0
    };
    pitch_gain = pitch_gain.min(gain);
    *period = (2 * best_period)
        .saturating_add_signed(offset)
        .max(min_period_original);
    pitch_gain
}

fn pitch_filter(
    spectrum: &mut [Complex32; FREQ_SIZE],
    pitch: &[Complex32; FREQ_SIZE],
    energy: &[f32; N_BANDS],
    pitch_energy: &[f32; N_BANDS],
    correlation: &[f32; N_BANDS],
    gains: &[f32; N_BANDS],
) {
    let mut ratio = [0.0; N_BANDS];
    for band in 0..N_BANDS {
        let value = if correlation[band] > gains[band] {
            1.0
        } else {
            let corr2 = correlation[band] * correlation[band];
            let gain2 = gains[band] * gains[band];
            (corr2 * (1.0 - gain2) / (0.001 + gain2 * (1.0 - corr2)))
                .clamp(0.0, 1.0)
                .sqrt()
        };
        ratio[band] = value * (energy[band] / (1e-8 + pitch_energy[band])).sqrt();
    }
    for (index, value) in interp_band_gain(&ratio).into_iter().enumerate() {
        spectrum[index] = spectrum[index] + pitch[index].scale(value);
    }
    let new_energy = compute_band_energy(spectrum);
    let normalization =
        std::array::from_fn(|band| (energy[band] / (1e-8 + new_energy[band])).sqrt());
    for (value, scale) in spectrum.iter_mut().zip(interp_band_gain(&normalization)) {
        *value = value.scale(scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_window_matches_upstream_table_anchors() {
        assert!((half_window_value(0) - 4.205_491_7e-6).abs() < 1e-12);
        assert_eq!(half_window_value(479), 1.0);
    }

    #[test]
    fn band_interpolation_leaves_above_20khz_zero() {
        let bands = [1.0; N_BANDS];
        let bins = interp_band_gain(&bands);
        assert!(bins[..400].iter().all(|value| *value == 1.0));
        assert!(bins[400..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn pitch_tracker_recovers_upstream_grid_tone() {
        let mut buffer = [0.0; PITCH_BUF_SIZE];
        for (index, sample) in buffer.iter_mut().enumerate() {
            *sample =
                (2.0 * std::f32::consts::PI * 200.0 * index as f32 / SAMPLE_RATE as f32).sin();
        }
        let downsampled = pitch_downsample(&buffer);
        let searched = pitch_search(
            &downsampled[PITCH_MAX_PERIOD / 2..],
            &downsampled,
            PITCH_FRAME_SIZE,
            PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD,
        );
        let mut period = PITCH_MAX_PERIOD - searched;
        let gain = remove_doubling(&downsampled, &mut period, 0, 0.0);
        assert!(period.abs_diff(240) <= 2, "period={period}");
        // The upstream pre-whitening FIR intentionally lowers the normalized
        // correlation of a bare sine; the v0.2 float path yields ~0.623.
        assert!(gain > 0.6, "gain={gain}");
    }
}
