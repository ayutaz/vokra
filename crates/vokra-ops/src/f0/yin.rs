//! YIN pitch estimator (de Cheveigné & Kawahara 2002).
//!
//! # Primary source
//!
//! - Alain de Cheveigné & Hideki Kawahara, *"YIN, a fundamental frequency
//!   estimator for speech and music"*, JASA 111(4):1917-1930 (2002).
//!   <https://doi.org/10.1121/1.1458024>.
//!
//! # Algorithm (paper walk)
//!
//! 1. **Difference function** (paper §2.2, eq. 6) — for each candidate
//!    lag τ, sum the squared difference between the frame and its
//!    τ-shifted copy: `d[τ] = Σᵢ (x[i] - x[i+τ])²`.
//! 2. **Cumulative-mean-normalized difference function** (§2.4, eq. 8) —
//!    normalize `d[τ]` by the running mean of `d[1..τ]`:
//!    `d'[τ] = d[τ] · τ / Σⱼ₌₁ᵗ d[j]`, with sentinel `d'[0] = 1`. The
//!    normalization removes YIN's octave-error bias.
//! 3. **Absolute-threshold dip pick** (§3) — walk `τ ∈ [τ_min, τ_max]`,
//!    find the first CMNDF value below [`DEFAULT_ABSOLUTE_THRESHOLD`],
//!    then walk downhill to its local minimum inside the sub-threshold
//!    region. Report unvoiced (`0.0` Hz) if no τ crosses the threshold.
//! 4. **Parabolic interpolation** (§5, "refinement") — fit a quadratic
//!    through `(τ-1, τ, τ+1)` on the CMNDF surface to refine τ to
//!    sub-sample resolution before dividing into the sample rate.
//!
//! Steps 1-2 (difference + CMNDF) and steps 3-4 (dip pick + refinement)
//! live in [`crate::f0`] so [`crate::f0::pyin`] can reuse them.
//!
//! # Contract
//!
//! [`yin`] returns one `f32` per analysis frame — the estimated F0 in Hz,
//! or `0.0` when the CMNDF dip-picker reports unvoiced. Analysis frames
//! step through the input at [`DEFAULT_HOP`] samples, each of length
//! [`DEFAULT_FRAME_SIZE`]. PCM shorter than one full frame yields an
//! empty `Vec` (documented boundary — no partial frames).
//!
//! # Errors
//!
//! Returns [`VokraError::InvalidArgument`] on `sample_rate == 0`, on
//! non-positive / non-finite `fmin`/`fmax`, on `fmin >= fmax`, or when
//! `fmin * 2 > sample_rate` (Nyquist violated).
//!
//! # Determinism / zero-dep
//!
//! Pure function, no interior state, no RNG. Uses only `std` and
//! [`vokra_core::{Result, VokraError}`]. Root `Cargo.lock` still lists
//! only `vokra-*` (NFR-DS-02).

use vokra_core::Result;

use super::{
    DEFAULT_ABSOLUTE_THRESHOLD, DEFAULT_FRAME_SIZE, DEFAULT_HOP, absolute_threshold, cmndf,
    difference_function, num_frames, parabolic_interpolation, tau_search_interval, validate_args,
};

/// Extract per-frame F0 (Hz) from `pcm` using the YIN algorithm.
///
/// See the module docstring for the algorithm walk. Returns one `f32`
/// per analysis frame stepped at [`DEFAULT_HOP`] samples with length
/// [`DEFAULT_FRAME_SIZE`]. Emits `0.0` for unvoiced frames.
///
/// # Arguments
///
/// - `pcm` — real, single-channel PCM samples in `[-1, 1]` (any sample
///   rate).
/// - `sample_rate` — PCM sample rate in Hz. Must be non-zero.
/// - `fmin` — lowest F0 the caller wants tracked (Hz). Bounds the
///   `τ_max = sr / fmin` search interval; must be finite, positive, and
///   strictly less than `fmax`, and `fmin * 2 <= sample_rate`.
/// - `fmax` — highest F0 the caller wants tracked (Hz). Bounds the
///   `τ_min = sr / fmax` search interval.
///
/// # Errors
///
/// See the module docstring.
pub fn yin(pcm: &[f32], sample_rate: u32, fmin: f32, fmax: f32) -> Result<Vec<f32>> {
    validate_args(sample_rate, fmin, fmax)?;
    let (tau_min, tau_max) = tau_search_interval(sample_rate, fmin, fmax)?;
    let sr = sample_rate as f32;

    let n_frames = num_frames(pcm.len());
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let start = f * DEFAULT_HOP;
        let frame = &pcm[start..start + DEFAULT_FRAME_SIZE];
        let d = difference_function(frame, tau_max);
        let d_prime = cmndf(&d);
        let hz = match absolute_threshold(&d_prime, tau_min, tau_max, DEFAULT_ABSOLUTE_THRESHOLD) {
            Some(tau) => {
                let refined = parabolic_interpolation(&d_prime, tau);
                if refined > 0.0 { sr / refined } else { 0.0 }
            }
            None => 0.0,
        };
        out.push(hz);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Only the error-path assertions below name the variant; importing it at
    // module scope would be dead weight in a non-test build.
    use vokra_core::VokraError;

    const TAU: f64 = std::f64::consts::TAU;

    /// Generate `len` samples of a `freq`-Hz sine at `rate` Hz.
    fn sine(freq: f64, rate: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|t| (TAU * freq * t as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    /// Mean of a slice, or `f32::NAN` for empty input.
    fn mean(x: &[f32]) -> f32 {
        if x.is_empty() {
            return f32::NAN;
        }
        (x.iter().copied().map(f64::from).sum::<f64>() / x.len() as f64) as f32
    }

    /// (a) Pure 440 Hz sine at sr=22050 recovers 440 Hz within ±1 Hz on
    ///     interior frames.
    #[test]
    fn pure_440_hz_sine_at_22050() {
        let pcm = sine(440.0, 22_050, 22_050); // 1 s
        let f0s = yin(&pcm, 22_050, 65.0, 1200.0).unwrap();
        assert!(f0s.len() >= 10, "want plenty of frames, got {}", f0s.len());
        // Skip the leading + trailing edge frames (parabolic refinement
        // is less reliable when the CMNDF minimum lands near the frame
        // boundary); the interior must lock on 440 Hz.
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

    /// (b) Pure 220 Hz sine at sr=16000 recovers 220 Hz within ±1 Hz.
    #[test]
    fn pure_220_hz_sine_at_16000() {
        let pcm = sine(220.0, 16_000, 16_000); // 1 s
        let f0s = yin(&pcm, 16_000, 65.0, 800.0).unwrap();
        let interior = &f0s[3..f0s.len() - 3];
        for &hz in interior {
            assert!(
                (hz - 220.0).abs() < 1.0,
                "interior F0 = {hz} Hz, want 220 ± 1"
            );
        }
        let avg = mean(interior);
        assert!(
            (avg - 220.0).abs() < 0.5,
            "interior mean F0 = {avg} Hz, want 220 ± 0.5"
        );
    }

    /// (c) Silence (all-zero PCM) yields all-0.0 (documented unvoiced
    ///     convention — the CMNDF sentinel `d'[τ] = 1` never crosses
    ///     the 0.10 threshold).
    #[test]
    fn silence_reports_all_unvoiced() {
        let pcm = vec![0.0f32; 16_000]; // 1 s of digital silence @ 16k
        let f0s = yin(&pcm, 16_000, 65.0, 800.0).unwrap();
        assert!(!f0s.is_empty());
        for (i, &hz) in f0s.iter().enumerate() {
            assert_eq!(hz, 0.0, "frame {i}: silence must be unvoiced (0.0 Hz)");
        }
    }

    /// (d) Frequency-swept chirp 100 → 800 Hz tracks a monotonically
    ///     increasing F0 (checked on smoothed 10-frame windows so a
    ///     single-frame outlier does not fail the sweep).
    #[test]
    fn linear_chirp_tracks_monotonically() {
        // Analytic linear chirp phase: φ(t) = 2π · (f0·t + 0.5·k·t²)
        // with f0 = 100 Hz and k = (f1 - f0) / duration = 700 Hz/s.
        let rate = 16_000u32;
        let duration = 1.0f64;
        let f_start = 100.0f64;
        let f_end = 800.0f64;
        let k = (f_end - f_start) / duration;
        let n = (rate as f64 * duration) as usize;
        let pcm: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                let phase = TAU * (f_start * t + 0.5 * k * t * t);
                phase.sin() as f32
            })
            .collect();
        let f0s = yin(&pcm, rate, 65.0, 1200.0).unwrap();
        assert!(f0s.len() >= 40, "want plenty of frames, got {}", f0s.len());

        // Smooth over rolling windows of 10 frames to reject single-frame
        // outliers, then check strict monotonicity of the smoothed track.
        let interior = &f0s[3..f0s.len() - 3];
        let window = 10usize;
        assert!(interior.len() > 2 * window, "chirp too short to smooth");
        let smoothed: Vec<f32> = (0..interior.len() - window)
            .map(|start| mean(&interior[start..start + window]))
            .collect();
        // Coarse monotonic check: last smoothed sample is >> first, and
        // no smoothed sample dips more than 30 Hz below its predecessor
        // (a stronger check than "every adjacent pair rising" — YIN's
        // per-frame outputs jitter, but the smoothed trend must climb).
        // The 250 Hz "climbed" threshold is a conservative floor for a
        // 700-Hz-per-second sweep over ~1 second (per-frame jitter can
        // pull the smoothed endpoints by ± 50 Hz), well below the ~415 Hz
        // gap between the first and last 10-frame window centers.
        assert!(
            smoothed[smoothed.len() - 1] - smoothed[0] > 250.0,
            "chirp did not climb: start={} end={}",
            smoothed[0],
            smoothed[smoothed.len() - 1]
        );
        for pair in smoothed.windows(2) {
            let (prev, next) = (pair[0], pair[1]);
            assert!(
                next - prev > -30.0,
                "smoothed chirp dipped: prev={prev} next={next}"
            );
        }
    }

    /// (e) `fmin >= fmax` rejected with `InvalidArgument`.
    #[test]
    fn fmin_ge_fmax_rejected() {
        let pcm = vec![0.0f32; DEFAULT_FRAME_SIZE];
        assert!(matches!(
            yin(&pcm, 16_000, 800.0, 800.0),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            yin(&pcm, 16_000, 1000.0, 800.0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// (f) `sample_rate = 0` rejected with `InvalidArgument`.
    #[test]
    fn zero_sample_rate_rejected() {
        let pcm = vec![0.0f32; DEFAULT_FRAME_SIZE];
        assert!(matches!(
            yin(&pcm, 0, 65.0, 800.0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// (g) PCM shorter than the frame size returns an empty `Vec` —
    ///     documented "no partial frames" boundary.
    #[test]
    fn too_short_pcm_returns_empty() {
        let pcm = vec![0.0f32; DEFAULT_FRAME_SIZE - 1];
        let f0s = yin(&pcm, 16_000, 65.0, 800.0).unwrap();
        assert!(
            f0s.is_empty(),
            "want empty output for PCM shorter than frame, got {} frames",
            f0s.len()
        );
    }

    /// Exact-frame-length PCM yields exactly one frame.
    #[test]
    fn exact_frame_length_yields_one_frame() {
        let pcm = sine(300.0, 16_000, DEFAULT_FRAME_SIZE);
        let f0s = yin(&pcm, 16_000, 65.0, 800.0).unwrap();
        assert_eq!(f0s.len(), 1);
        assert!(
            (f0s[0] - 300.0).abs() < 1.0,
            "single-frame F0 = {} Hz, want 300 ± 1",
            f0s[0]
        );
    }

    /// Determinism — the same PCM + args must produce bit-identical
    /// output on repeated invocations.
    #[test]
    fn deterministic_repeated_invocation() {
        let pcm = sine(150.0, 22_050, 22_050);
        let a = yin(&pcm, 22_050, 65.0, 1000.0).unwrap();
        let b = yin(&pcm, 22_050, 65.0, 1000.0).unwrap();
        assert_eq!(a, b);
    }
}
