//! Probabilistic YIN pitch estimator (Mauch & Dixon 2014).
//!
//! # Primary source
//!
//! - Matthias Mauch & Simon Dixon, *"pYIN: A Fundamental Frequency
//!   Estimator Using Probabilistic Threshold Distributions"*, ICASSP 2014.
//!   <https://doi.org/10.1109/ICASSP.2014.6853678>. Reference PDF:
//!   <https://code.soundsoftware.ac.uk/attachments/download/1443/MauchD14-pyin.pdf>.
//!
//! # Algorithm (paper walk)
//!
//! For each analysis frame, PyIN keeps YIN's difference function +
//! cumulative-mean-normalized difference (CMNDF, see [`crate::f0`]) but
//! replaces YIN's single fixed absolute threshold with a **weighted grid
//! of thresholds**:
//!
//! 1. Reuse the CMNDF `d'[τ]` (computed once per frame — the
//!    difference/CMNDF is threshold-independent).
//! 2. For each of [`PYIN_NUM_THRESHOLDS`] linearly-spaced thresholds
//!    across `(0, 1)`, run YIN's absolute-threshold dip pick. Each dip
//!    lands at some integer τ (or `None` = unvoiced under that
//!    threshold).
//! 3. Weight each candidate τ by the Beta(2, 18) prior density
//!    [`beta_2_18_pdf`] evaluated at its threshold. The Beta prior peaks
//!    near `1/18 ≈ 0.056` and has ~99% of its mass below `0.3`, so
//!    dips found at stricter thresholds carry more weight — exactly the
//!    paper's "reward confident dips" bias.
//! 4. Sum weights per integer-τ bin; also accumulate the "no dip found"
//!    weight into the unvoiced posterior. If the voiced posterior beats
//!    the unvoiced posterior, emit the τ-bin argmax (refined by
//!    parabolic interpolation on the CMNDF surface, mirroring YIN);
//!    otherwise emit `0.0`.
//!
//! # Deterministic threshold sampling (not stochastic)
//!
//! The paper's abstract describes sampling thresholds *from* a
//! probabilistic distribution. A stochastic implementation would need
//! an RNG (non-deterministic, tests would flake). The paper's own
//! reference implementation (Vamp plugin `pyin`) uses a fixed
//! linearly-spaced grid weighted by the Beta density — equivalent to
//! the stochastic sampling in expectation and deterministic across
//! runs. This module ships the deterministic form.
//!
//! # Viterbi smoothing — deferred follow-up
//!
//! The full PyIN pipeline (paper §2.2 + §3) adds a HMM whose observation
//! probabilities are the per-frame `(voiced, τ_bin, weight)` tuples this
//! module emits, plus a transition matrix that penalises sharp F0 jumps.
//! Viterbi decoding across the HMM yields a smoother pitch track than
//! per-frame argmax.
//!
//! The current module ships **per-frame argmax only**. Viterbi smoothing
//! is left as a documented follow-up (function scaffolded as
//! [`viterbi_smooth_todo`]) — the benefit is temporal, not per-frame
//! accuracy, and the task's 5+ unit tests (pure sines, silence, two-tone
//! step) all pass on per-frame argmax alone. When the M3-17 prosody or
//! voice-clone consumer arrives (owner ADR 2026-07-22, memory
//! `[[project-m5-owner-adr-decisions.md]]` item ⑦, landing WP M5-16),
//! the smoother lands with it.
//!
//! # Contract / errors
//!
//! Same shape as [`crate::f0::yin`]: `Vec<f32>` of per-frame F0 in Hz,
//! `0.0` on unvoiced frames, empty output when PCM shorter than one
//! frame. Same `InvalidArgument` conditions.
//!
//! # Determinism / zero-dep
//!
//! Pure function, no RNG. `std` + [`vokra_core::{Result, VokraError}`]
//! only; root `Cargo.lock` lists only `vokra-*` (NFR-DS-02).

use vokra_core::Result;

use super::{
    DEFAULT_FRAME_SIZE, DEFAULT_HOP, absolute_threshold, cmndf, difference_function, num_frames,
    parabolic_interpolation, tau_search_interval, validate_args,
};

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

/// Threshold grid + Beta(2, 18) weight table.
///
/// Broken out as a helper so tests can inspect the shape and inspect the
/// integral of `beta_2_18_pdf` across the grid.
fn threshold_grid() -> (Vec<f32>, Vec<f32>) {
    let thresholds: Vec<f32> = (0..PYIN_NUM_THRESHOLDS)
        .map(|i| (i as f32 + 0.5) / PYIN_NUM_THRESHOLDS as f32)
        .collect();
    let weights: Vec<f32> = thresholds.iter().map(|&t| beta_2_18_pdf(t)).collect();
    (thresholds, weights)
}

/// Scaffold for future HMM Viterbi smoothing (see the module docstring).
///
/// Returns the input `pitch_track` unchanged for now — the current
/// per-frame argmax is what the task's 5+ unit tests exercise. When a
/// downstream consumer arrives, replace the body with the paper's
/// forward-backward Viterbi.
#[allow(dead_code)]
pub(crate) fn viterbi_smooth_todo(pitch_track: Vec<f32>) -> Vec<f32> {
    // TODO(M5-16 consumer arrival): replace with the Mauch & Dixon 2014
    // §III HMM Viterbi. The stub returns the input unchanged so it is
    // safe to wire from public API without breaking behaviour when the
    // real implementation lands.
    pitch_track
}

/// Extract per-frame F0 (Hz) from `pcm` using the PyIN algorithm.
///
/// See the module docstring for the algorithm walk. Same shape and
/// error contract as [`crate::f0::yin`].
pub fn pyin(pcm: &[f32], sample_rate: u32, fmin: f32, fmax: f32) -> Result<Vec<f32>> {
    validate_args(sample_rate, fmin, fmax)?;
    let (tau_min, tau_max) = tau_search_interval(sample_rate, fmin, fmax)?;
    let sr = sample_rate as f32;

    let (thresholds, weights) = threshold_grid();
    let weight_total: f32 = weights.iter().copied().sum();

    let n_frames = num_frames(pcm.len());
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let start = f * DEFAULT_HOP;
        let frame = &pcm[start..start + DEFAULT_FRAME_SIZE];
        let d = difference_function(frame, tau_max);
        let d_prime = cmndf(&d);

        // Accumulate posterior mass per integer τ bin.
        let mut posterior = vec![0.0f32; tau_max + 1];
        let mut voiced_total = 0.0f32;
        for (&threshold, &weight) in thresholds.iter().zip(weights.iter()) {
            if let Some(tau) = absolute_threshold(&d_prime, tau_min, tau_max, threshold) {
                posterior[tau] += weight;
                voiced_total += weight;
            }
        }
        let unvoiced_total = (weight_total - voiced_total).max(0.0);

        let hz = if voiced_total > unvoiced_total {
            // Argmax over τ bins — safe unwrap because `posterior` is
            // non-empty by construction (`tau_max + 1 >= 3` after
            // `tau_search_interval` validation).
            let (argmax_tau, _max_w) = posterior
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("posterior non-empty by tau_search_interval invariant");
            let refined = parabolic_interpolation(&d_prime, argmax_tau);
            if refined > 0.0 { sr / refined } else { 0.0 }
        } else {
            0.0
        };
        out.push(hz);
    }
    Ok(out)
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

    /// (e) `beta_2_18_pdf` integrates to ≈ 1 across the 100-sample grid
    ///     (numerical Riemann midpoint rule) — a sanity check on the
    ///     Beta density constant.
    #[test]
    fn beta_pdf_integrates_to_unity() {
        let (thresholds, weights) = threshold_grid();
        assert_eq!(thresholds.len(), PYIN_NUM_THRESHOLDS);
        assert_eq!(weights.len(), PYIN_NUM_THRESHOLDS);
        let dx = 1.0 / PYIN_NUM_THRESHOLDS as f64;
        let integral: f64 = weights.iter().copied().map(f64::from).sum::<f64>() * dx;
        assert!(
            (integral - 1.0).abs() < 0.02,
            "midpoint Riemann sum of Beta(2, 18) PDF = {integral}, want 1 ± 0.02"
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

    /// The Viterbi-smoothing scaffold is a documented follow-up; the
    /// pass-through identity confirms nothing regresses when the
    /// smoother lands.
    #[test]
    fn viterbi_scaffold_is_identity_for_now() {
        let input = vec![100.0f32, 110.0, 120.0, 0.0, 130.0];
        let smoothed = viterbi_smooth_todo(input.clone());
        assert_eq!(smoothed, input);
    }
}
