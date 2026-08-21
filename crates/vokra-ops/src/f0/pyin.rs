//! Probabilistic YIN pitch estimator (Mauch & Dixon 2014).
//!
//! # Primary source
//!
//! - Matthias Mauch & Simon Dixon, *"pYIN: A Fundamental Frequency
//!   Estimator Using Probabilistic Threshold Distributions"*, ICASSP 2014.
//!   <https://doi.org/10.1109/ICASSP.2014.6853678>. Reference PDF:
//!   <https://code.soundsoftware.ac.uk/attachments/download/1443/MauchD14-pyin.pdf>.
//!
//! # Algorithm
//!
//! This is the complete deterministic PyIN pipeline, including temporal
//! smoothing. For each frame it computes the YIN CMNDF, retains every local
//! trough, integrates each trough's probability over 100 Beta(2, 18)
//! threshold intervals, and maps the candidates into 0.1-semitone bins. A
//! two-layer (voiced/unvoiced) hidden Markov model then constrains pitch jumps
//! and Viterbi-decodes the whole track. The HMM parameters follow the
//! canonical PyIN defaults: 35.92 octaves/s maximum transition rate, 0.01
//! voiced-state switch probability, and a Boltzmann(2) prior favouring shorter
//! candidate periods.
//!
//! The implementation was independently checked against librosa 0.11.0's
//! PyIN implementation at tag commit
//! `af8c839fb15317fa2712ea66e7a22da6a9267b32`; librosa is an offline parity
//! oracle only and is not linked or vendored into Vokra.
//!
//! # Contract / errors
//!
//! [`pyin`] keeps the historical shape shared with [`crate::f0::yin`]:
//! `Vec<f32>` of per-frame F0 in Hz, `0.0` on unvoiced frames. The richer
//! [`pyin_detailed`] result also reports the Viterbi voiced flag and the
//! pre-decode voiced probability, which lets the CLI report real confidence
//! instead of a fabricated binary value.
//!
//! # Determinism / zero-dep
//!
//! Pure function, no RNG. `std` + [`vokra_core::{Result, VokraError}`]
//! only; root `Cargo.lock` lists only `vokra-*` (NFR-DS-02).

use vokra_core::Result;

use super::{DEFAULT_FRAME_SIZE, DEFAULT_HOP, num_frames, tau_search_interval, validate_args};

/// Number of threshold samples the PyIN posterior sums over.
///
/// `100` matches the paper's Vamp reference (Mauch & Dixon 2014 §II.C).
/// Larger values sharpen the posterior at higher cost; smaller values
/// weaken the Beta-weighted vote.
pub const PYIN_NUM_THRESHOLDS: usize = 100;

/// Beta(2, 18) probability density function evaluated at `x`.
///
/// Beta(α, β) PDF at parameters `α = 2`, `β = 18`:
///
/// ```text
///   f(x) = x^(α-1) · (1-x)^(β-1) / B(α, β)
///        = x · (1-x)^17 / B(2, 18)
///   B(2, 18) = Γ(2)·Γ(18) / Γ(20) = 1! · 17! / 19! = 1 / (18 · 19) = 1/342
/// ```
///
/// so `f(x) = 342 · x · (1-x)^17`. Peak (mode) at `1/(β+1-1) = 1/18 ≈ 0.056`;
/// mean at `α/(α+β) = 0.10`. Returns `0.0` outside the open interval
/// `(0, 1)`.
///
/// `(1-x)^17` is computed by repeated squaring (5 multiplies) instead of
/// [`f32::powi`] to keep the constant deterministic across libm variants
/// and avoid a hidden Rust intrinsic that may resolve differently on
/// SIMD kernels — the `powi` result varies by up to a few ULPs on some
/// platforms; the closed-form multiply chain matches exactly.
pub fn beta_2_18_pdf(x: f32) -> f32 {
    if !(x > 0.0 && x < 1.0) {
        return 0.0;
    }
    let a = 1.0 - x;
    let a2 = a * a;
    let a4 = a2 * a2;
    let a8 = a4 * a4;
    let a16 = a8 * a8;
    let a17 = a16 * a;
    342.0 * x * a17
}

/// Closed-form Beta(2, 18) cumulative distribution.
fn beta_2_18_cdf(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        1.0 - (1.0 - x).powi(18) * (1.0 + 18.0 * x)
    }
}

/// The 100 exact probability masses between the 101 evenly-spaced threshold
/// edges. Using CDF differences is materially different from sampling the PDF
/// at midpoints, especially in the high-density low-threshold region.
fn threshold_masses() -> Vec<f64> {
    (0..PYIN_NUM_THRESHOLDS)
        .map(|i| {
            let lo = i as f64 / PYIN_NUM_THRESHOLDS as f64;
            let hi = (i + 1) as f64 / PYIN_NUM_THRESHOLDS as f64;
            beta_2_18_cdf(hi) - beta_2_18_cdf(lo)
        })
        .collect()
}

const BOLTZMANN_PARAMETER: f64 = 2.0;
const NO_TROUGH_PROB: f64 = 0.01;
const RESOLUTION_SEMITONES: f64 = 0.1;
const MAX_TRANSITION_RATE_OCTAVES_PER_SEC: f64 = 35.92;
const SWITCH_PROB: f64 = 0.01;

/// One PyIN output frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PyinFrame {
    /// Fundamental frequency in Hz, or `0.0` for an unvoiced Viterbi state.
    pub hz: f32,
    /// Whether Viterbi selected the voiced layer of the HMM.
    pub voiced: bool,
    /// Marginal voiced observation probability before temporal decoding.
    pub confidence: f32,
}

#[derive(Debug)]
struct Observation {
    probabilities: Vec<f64>,
    voiced_probability: f64,
}

/// Librosa's canonical PyIN difference-function form. This is kept separate
/// from the public YIN primitive because PyIN's reference pipeline uses a
/// fixed-frame autocorrelation identity rather than the pair-count form used
/// by [`super::yin`].
fn pyin_cmndf(frame: &[f32], min_period: usize, max_period: usize) -> Vec<f64> {
    let energy: f64 = frame.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let mut prefix_energy = 0.0f64;
    let mut differences = vec![0.0f64; max_period + 1];
    for period in 1..=max_period {
        let x = f64::from(frame[period - 1]);
        prefix_energy += x * x;
        let autocorrelation: f64 = frame[..frame.len() - period]
            .iter()
            .zip(frame[period..].iter())
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        differences[period] = (2.0 * (energy - autocorrelation) - prefix_energy).max(0.0);
    }

    let mut cumulative = 0.0f64;
    let mut out = Vec::with_capacity(max_period - min_period + 1);
    for period in 1..=max_period {
        cumulative += differences[period];
        if period >= min_period {
            let mean = cumulative / period as f64;
            out.push(differences[period] / (mean + f64::MIN_POSITIVE));
        }
    }
    out
}

fn parabolic_shifts(values: &[f64]) -> Vec<f64> {
    let mut shifts = vec![0.0; values.len()];
    for i in 1..values.len().saturating_sub(1) {
        let a = values[i + 1] + values[i - 1] - 2.0 * values[i];
        let b = (values[i + 1] - values[i - 1]) / 2.0;
        if b.abs() < a.abs() {
            shifts[i] = -b / a;
        }
    }
    shifts
}

fn local_troughs(values: &[f64]) -> Vec<usize> {
    if values.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::new();
    if values[0] < values[1] {
        out.push(0);
    }
    for i in 1..values.len() - 1 {
        if values[i] < values[i - 1] && values[i] <= values[i + 1] {
            out.push(i);
        }
    }
    if values[values.len() - 1] < values[values.len() - 2] {
        out.push(values.len() - 1);
    }
    out
}

fn boltzmann_probability(rank: usize, count: usize) -> f64 {
    debug_assert!(rank < count);
    let numerator =
        (1.0 - (-BOLTZMANN_PARAMETER).exp()) * (-BOLTZMANN_PARAMETER * rank as f64).exp();
    numerator / (1.0 - (-BOLTZMANN_PARAMETER * count as f64).exp())
}

fn frame_observation(
    frame: &[f32],
    sample_rate: f64,
    min_period: usize,
    max_period: usize,
    fmin: f64,
    n_pitch_bins: usize,
    n_bins_per_semitone: usize,
    beta_masses: &[f64],
) -> Observation {
    let yin = pyin_cmndf(frame, min_period, max_period);
    let shifts = parabolic_shifts(&yin);
    let troughs = local_troughs(&yin);
    let mut voiced = vec![0.0f64; n_pitch_bins];

    if !troughs.is_empty() {
        let global_min = *troughs
            .iter()
            .min_by(|&&a, &&b| yin[a].total_cmp(&yin[b]))
            .expect("non-empty trough list");
        let mut trough_probabilities = vec![0.0f64; troughs.len()];

        for (threshold_index, &threshold_mass) in beta_masses.iter().enumerate() {
            let threshold = (threshold_index + 1) as f64 / PYIN_NUM_THRESHOLDS as f64;
            let qualifying: Vec<usize> = troughs
                .iter()
                .enumerate()
                .filter_map(|(rank, &trough)| (yin[trough] < threshold).then_some(rank))
                .collect();
            if qualifying.is_empty() {
                let index = troughs
                    .iter()
                    .position(|&trough| trough == global_min)
                    .expect("global minimum is a trough");
                trough_probabilities[index] += NO_TROUGH_PROB * threshold_mass;
            } else {
                for (rank, &trough_list_index) in qualifying.iter().enumerate() {
                    trough_probabilities[trough_list_index] +=
                        boltzmann_probability(rank, qualifying.len()) * threshold_mass;
                }
            }
        }

        // Match the canonical reference's indexed assignment: if two periods
        // quantize to the same pitch bin, the later (longer-period) trough
        // replaces the earlier one rather than double-counting probability.
        for (&trough, &probability) in troughs.iter().zip(trough_probabilities.iter()) {
            if probability == 0.0 {
                continue;
            }
            let period = min_period as f64 + trough as f64 + shifts[trough];
            let frequency = sample_rate / period;
            let raw_bin = 12.0 * n_bins_per_semitone as f64 * (frequency / fmin).log2();
            let bin = raw_bin.round().clamp(0.0, (n_pitch_bins - 1) as f64) as usize;
            voiced[bin] = probability;
        }
    }

    let voiced_probability = voiced.iter().sum::<f64>().clamp(0.0, 1.0);
    let unvoiced_probability = (1.0 - voiced_probability) / n_pitch_bins as f64;
    let mut probabilities = voiced;
    probabilities.extend(std::iter::repeat_n(unvoiced_probability, n_pitch_bins));
    Observation {
        probabilities,
        voiced_probability,
    }
}

fn transition_radius(sample_rate: u32, n_bins_per_semitone: usize) -> usize {
    let semitones_per_frame = (MAX_TRANSITION_RATE_OCTAVES_PER_SEC * 12.0 * DEFAULT_HOP as f64
        / sample_rate as f64)
        .round() as usize;
    // The canonical triangular transition width is `2*radius+1` in effect:
    // for the configured `width = semitones*bins + 1`, width is always odd.
    (semitones_per_frame * n_bins_per_semitone) / 2
}

fn viterbi_decode(observations: &[Observation], n_pitch_bins: usize, radius: usize) -> Vec<usize> {
    if observations.is_empty() {
        return Vec::new();
    }
    let n_states = 2 * n_pitch_bins;
    let tiny = f64::MIN_POSITIVE;
    let initial = -(n_states as f64).ln();
    let mut values: Vec<f64> = observations[0]
        .probabilities
        .iter()
        .map(|&p| (p + tiny).ln() + initial)
        .collect();
    let mut pointers = vec![vec![0usize; n_states]; observations.len()];

    // Each source row of the truncated triangular pitch transition has a
    // different normalizer near the fmin/fmax edges.
    let row_normalizers: Vec<f64> = (0..n_pitch_bins)
        .map(|source| {
            let lo = source.saturating_sub(radius);
            let hi = (source + radius).min(n_pitch_bins - 1);
            (lo..=hi)
                .map(|dest| (radius + 1 - source.abs_diff(dest)) as f64)
                .sum()
        })
        .collect();

    for (time, observation) in observations.iter().enumerate().skip(1) {
        let mut next = vec![f64::NEG_INFINITY; n_states];
        for dest_voice in 0..2usize {
            for dest_pitch in 0..n_pitch_bins {
                let dest_state = dest_voice * n_pitch_bins + dest_pitch;
                let source_lo = dest_pitch.saturating_sub(radius);
                let source_hi = (dest_pitch + radius).min(n_pitch_bins - 1);
                let mut best_value = f64::NEG_INFINITY;
                let mut best_source = 0usize;
                for source_voice in 0..2usize {
                    let switch = if source_voice == dest_voice {
                        1.0 - SWITCH_PROB
                    } else {
                        SWITCH_PROB
                    };
                    for source_pitch in source_lo..=source_hi {
                        let triangle = (radius + 1 - source_pitch.abs_diff(dest_pitch)) as f64;
                        let transition = switch * triangle / row_normalizers[source_pitch];
                        let source_state = source_voice * n_pitch_bins + source_pitch;
                        let candidate = values[source_state] + (transition + tiny).ln();
                        if candidate > best_value {
                            best_value = candidate;
                            best_source = source_state;
                        }
                    }
                }
                next[dest_state] = best_value + (observation.probabilities[dest_state] + tiny).ln();
                pointers[time][dest_state] = best_source;
            }
        }
        values = next;
    }

    let mut states = vec![0usize; observations.len()];
    states[observations.len() - 1] = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .expect("non-empty state vector");
    for time in (0..observations.len() - 1).rev() {
        states[time] = pointers[time + 1][states[time + 1]];
    }
    states
}

/// Extract per-frame F0 (Hz) from `pcm` using the PyIN algorithm.
///
/// See the module docstring for the algorithm walk. Same shape and
/// error contract as [`crate::f0::yin`].
pub fn pyin(pcm: &[f32], sample_rate: u32, fmin: f32, fmax: f32) -> Result<Vec<f32>> {
    Ok(pyin_detailed(pcm, sample_rate, fmin, fmax)?
        .into_iter()
        .map(|frame| frame.hz)
        .collect())
}

/// Extract temporally-smoothed F0 together with voiced state and confidence.
///
/// The confidence is the per-frame voiced observation probability before
/// Viterbi smoothing. It remains meaningful on unvoiced frames (where values
/// near zero are strong evidence for unvoiced audio).
pub fn pyin_detailed(
    pcm: &[f32],
    sample_rate: u32,
    fmin: f32,
    fmax: f32,
) -> Result<Vec<PyinFrame>> {
    validate_args(sample_rate, fmin, fmax)?;
    let (tau_min, tau_max) = tau_search_interval(sample_rate, fmin, fmax)?;
    let n_frames = num_frames(pcm.len());
    if n_frames == 0 {
        return Ok(Vec::new());
    }
    let n_bins_per_semitone = (1.0 / RESOLUTION_SEMITONES).ceil() as usize;
    let n_pitch_bins =
        (12.0 * n_bins_per_semitone as f64 * (f64::from(fmax) / f64::from(fmin)).log2()).floor()
            as usize
            + 1;
    let beta_masses = threshold_masses();
    let observations: Vec<Observation> = (0..n_frames)
        .map(|frame_index| {
            let start = frame_index * DEFAULT_HOP;
            frame_observation(
                &pcm[start..start + DEFAULT_FRAME_SIZE],
                sample_rate as f64,
                tau_min,
                tau_max,
                fmin as f64,
                n_pitch_bins,
                n_bins_per_semitone,
                &beta_masses,
            )
        })
        .collect();
    let states = viterbi_decode(
        &observations,
        n_pitch_bins,
        transition_radius(sample_rate, n_bins_per_semitone),
    );
    Ok(states
        .into_iter()
        .zip(observations)
        .map(|(state, observation)| {
            let voiced = state < n_pitch_bins;
            let pitch_bin = state % n_pitch_bins;
            let hz = if voiced {
                f64::from(fmin)
                    * 2.0f64.powf(pitch_bin as f64 / (12.0 * n_bins_per_semitone as f64))
            } else {
                0.0
            };
            PyinFrame {
                hz: hz as f32,
                voiced,
                confidence: observation.voiced_probability as f32,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f0::yin;
    use vokra_core::VokraError;

    const TAU: f64 = std::f64::consts::TAU;

    fn sine(freq: f64, rate: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|t| (TAU * freq * t as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    fn mean(x: &[f32]) -> f32 {
        if x.is_empty() {
            return f32::NAN;
        }
        (x.iter().copied().map(f64::from).sum::<f64>() / x.len() as f64) as f32
    }

    /// (a) Pure 440 Hz sine at sr=22050 → per-frame 440 Hz ± 1 Hz.
    #[test]
    fn pure_440_hz_sine_at_22050() {
        let pcm = sine(440.0, 22_050, 22_050);
        let f0s = pyin(&pcm, 22_050, 65.0, 1200.0).unwrap();
        let interior = &f0s[3..f0s.len() - 3];
        for &hz in interior {
            assert!(
                (hz - 440.0).abs() < 1.0,
                "interior F0 = {hz} Hz, want 440 ± 1"
            );
        }
        let avg = mean(interior);
        assert!(
            (avg - 440.0).abs() < 0.5,
            "interior mean F0 = {avg} Hz, want 440 ± 0.5"
        );
    }

    /// (b) Pure 110 Hz sine at sr=16000 → 110 Hz ± 1 Hz.
    #[test]
    fn pure_110_hz_sine_at_16000() {
        let pcm = sine(110.0, 16_000, 16_000);
        let f0s = pyin(&pcm, 16_000, 65.0, 800.0).unwrap();
        let interior = &f0s[3..f0s.len() - 3];
        for &hz in interior {
            assert!(
                (hz - 110.0).abs() < 1.0,
                "interior F0 = {hz} Hz, want 110 ± 1"
            );
        }
        let avg = mean(interior);
        assert!(
            (avg - 110.0).abs() < 0.5,
            "interior mean F0 = {avg} Hz, want 110 ± 0.5"
        );
    }

    /// (c) Silence → all-0.0 — the CMNDF sentinel never crosses any
    ///     threshold, so the unvoiced posterior wins on every frame.
    #[test]
    fn silence_reports_all_unvoiced() {
        let pcm = vec![0.0f32; 16_000];
        let f0s = pyin(&pcm, 16_000, 65.0, 800.0).unwrap();
        assert!(!f0s.is_empty());
        for (i, &hz) in f0s.iter().enumerate() {
            assert_eq!(hz, 0.0, "frame {i}: silence must be unvoiced (0.0 Hz)");
        }
    }

    /// (d) Two-tone step (0.5 s @ 220 Hz then 0.5 s @ 440 Hz) — first
    ///     "well inside" chunk tracks 220, second tracks 440. Skip the
    ///     transitional frames whose window straddles the step.
    #[test]
    fn two_tone_step_tracks_each_half() {
        let rate = 22_050u32;
        let half_len = rate as usize / 2; // 0.5 s
        let mut pcm = sine(220.0, rate, half_len);
        pcm.extend_from_slice(&sine(440.0, rate, half_len));

        let f0s = pyin(&pcm, rate, 65.0, 1200.0).unwrap();
        // (22050 - 2048) / 256 + 1 = 79 frames for a 1s signal at
        // sample_rate=22050, DEFAULT_FRAME_SIZE=2048, DEFAULT_HOP=256.
        assert!(f0s.len() >= 79, "want plenty of frames, got {}", f0s.len());

        // The step lands at PCM sample `half_len`. A frame starting at
        // `start` fully covers the "before" tone iff
        // `start + DEFAULT_FRAME_SIZE <= half_len`. Fully covers the
        // "after" tone iff `start >= half_len`.
        let mut before: Vec<f32> = Vec::new();
        let mut after: Vec<f32> = Vec::new();
        for (i, &hz) in f0s.iter().enumerate() {
            let start = i * DEFAULT_HOP;
            if start + DEFAULT_FRAME_SIZE <= half_len {
                before.push(hz);
            } else if start >= half_len {
                after.push(hz);
            }
        }
        assert!(before.len() >= 20, "want frames fully in 220 Hz half");
        assert!(after.len() >= 20, "want frames fully in 440 Hz half");
        for &hz in &before {
            assert!((hz - 220.0).abs() < 2.0, "220 Hz half yielded F0 = {hz} Hz");
        }
        for &hz in &after {
            assert!((hz - 440.0).abs() < 2.0, "440 Hz half yielded F0 = {hz} Hz");
        }
    }

    /// (e) Exact Beta CDF interval masses sum to one.
    #[test]
    fn beta_threshold_masses_sum_to_unity() {
        let weights = threshold_masses();
        assert_eq!(weights.len(), PYIN_NUM_THRESHOLDS);
        let integral: f64 = weights.iter().sum();
        assert!(
            (integral - 1.0).abs() < 1e-12,
            "Beta(2, 18) interval mass = {integral}, want 1"
        );
        // Beta PDF is zero outside (0, 1).
        assert_eq!(beta_2_18_pdf(0.0), 0.0);
        assert_eq!(beta_2_18_pdf(1.0), 0.0);
        assert_eq!(beta_2_18_pdf(-0.1), 0.0);
        assert_eq!(beta_2_18_pdf(1.1), 0.0);
        assert!(beta_2_18_pdf(0.05) > 0.0);
    }

    /// (f) `fmin >= fmax` rejected with `InvalidArgument`.
    #[test]
    fn fmin_ge_fmax_rejected() {
        let pcm = vec![0.0f32; DEFAULT_FRAME_SIZE];
        assert!(matches!(
            pyin(&pcm, 16_000, 800.0, 800.0),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            pyin(&pcm, 16_000, 1000.0, 800.0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// (g) YIN-vs-PyIN parity on 440 Hz sine — per-frame argmax matches
    ///     YIN's dip pick within ±2 Hz. This documents that PyIN
    ///     reduces to YIN when the CMNDF has a clean, single dip: every
    ///     threshold selects the same τ.
    #[test]
    fn pyin_matches_yin_within_2hz_on_pure_sine() {
        let pcm = sine(440.0, 22_050, 22_050);
        let y = yin(&pcm, 22_050, 65.0, 1200.0).unwrap();
        let p = pyin(&pcm, 22_050, 65.0, 1200.0).unwrap();
        assert_eq!(y.len(), p.len(), "frame counts must match");
        for (i, (&hy, &hp)) in y.iter().zip(p.iter()).enumerate().skip(3).take(y.len() - 6) {
            assert!(
                (hy - hp).abs() < 2.0,
                "frame {i}: yin={hy} Hz vs pyin={hp} Hz"
            );
        }
    }

    /// Zero sample_rate rejected with `InvalidArgument`.
    #[test]
    fn zero_sample_rate_rejected() {
        let pcm = vec![0.0f32; DEFAULT_FRAME_SIZE];
        assert!(matches!(
            pyin(&pcm, 0, 65.0, 800.0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// PCM shorter than one frame returns an empty `Vec` — same
    /// documented boundary as [`crate::f0::yin`].
    #[test]
    fn too_short_pcm_returns_empty() {
        let pcm = vec![0.0f32; DEFAULT_FRAME_SIZE - 1];
        let f0s = pyin(&pcm, 16_000, 65.0, 800.0).unwrap();
        assert!(f0s.is_empty());
    }

    /// Determinism — no RNG anywhere; identical input must produce
    /// bit-identical output.
    #[test]
    fn deterministic_repeated_invocation() {
        let pcm = sine(200.0, 22_050, 22_050);
        let a = pyin(&pcm, 22_050, 65.0, 1000.0).unwrap();
        let b = pyin(&pcm, 22_050, 65.0, 1000.0).unwrap();
        assert_eq!(a, b);
    }

    /// One isolated octave-scale observation cannot bypass the local pitch
    /// transition band, so Viterbi suppresses it.
    #[test]
    fn viterbi_suppresses_isolated_large_pitch_spike() {
        let n_pitch_bins = 80usize;
        let observations: Vec<Observation> = (0..7)
            .map(|time| {
                let mut probabilities = vec![1e-12; 2 * n_pitch_bins];
                probabilities[if time == 3 { 60 } else { 20 }] = 0.99;
                Observation {
                    probabilities,
                    voiced_probability: 0.99,
                }
            })
            .collect();
        let states = viterbi_decode(&observations, n_pitch_bins, 5);
        assert_eq!(states, vec![20; 7]);
    }

    #[test]
    fn detailed_silence_reports_real_zero_confidence() {
        let frames = pyin_detailed(&vec![0.0; 16_000], 16_000, 65.0, 800.0).unwrap();
        assert!(!frames.is_empty());
        for frame in frames {
            assert_eq!(frame.hz, 0.0);
            assert!(!frame.voiced);
            assert_eq!(frame.confidence, 0.0);
        }
    }
}
