//! Classical DSP F0 (fundamental frequency / pitch) extractors
//! (FR-OP-83 classical branch).
//!
//! # Reachability (was: "landing WP M5-16")
//!
//! This header carried a blanket "landing WP **M5-16**" label until
//! 2026-08-16, by which point it was half false. [`yin`] and [`pyin`] are
//! reachable from a shipped binary as `vokra-cli f0 --algo yin|pyin`; what
//! remains dormant under M5-16 is the UNIFIED `f0_extract` API of FR-OP-83
//! that would put these two and the neural members
//! (`vokra_models::f0::{rmvpe, fcpe, crepe}`) behind one entry point.
//!
//! The distinction is worth keeping straight: the two here need no
//! checkpoint, no license class and no `docs/license-audit.md` §3.1 row, so
//! nothing external gates them — they were simply never wired to a binary,
//! which is a different problem from being blocked. The neural members do
//! load weights and stay on `vokra-cli run --model <gguf>`.
//!
//! # Scope
//!
//! Weight-free, algorithm-only F0 estimators lifted from the primary
//! literature:
//!
//! - [`yin`] — YIN (de Cheveigné & Kawahara 2002, *"YIN, a fundamental
//!   frequency estimator for speech and music"*, JASA 111(4):1917-1930).
//!   Autocorrelation-in-time difference function + cumulative-mean-
//!   normalized difference (CMNDF) + absolute-threshold dip pick +
//!   parabolic interpolation.
//! - [`pyin`] — PyIN (Mauch & Dixon 2014, *"pYIN: A Fundamental Frequency
//!   Estimator Using Probabilistic Threshold Distributions"*, ICASSP 2014).
//!   All local CMNDF troughs are integrated over 100 Beta(2, 18) threshold
//!   intervals and decoded by the voiced/unvoiced pitch-bin HMM with Viterbi
//!   temporal smoothing. [`pyin_detailed`] additionally exposes voiced state
//!   and the real per-frame voiced probability.
//! - Harvest (WORLD vocoder) — deferred follow-up wave (task allows).
//!
//! # Placement rationale (2026-08-14 audit follow-up Wave 7)
//!
//! The neural F0 branch (RMVPE / FCPE / CREPE) lives in
//! [`vokra_models::f0`](../../vokra_models/f0/index.html) because it
//! loads per-model GGUF weight bundles. The classical branch does not
//! — it is a pure DSP algorithm parameterised only by
//! `(sample_rate, fmin, fmax)` — so it belongs one level lower, in the
//! [`vokra_ops`](crate) audio-dialect crate alongside [`crate::resample`]
//! / [`crate::agc`] / [`crate::hpf`]. Together the two branches implement
//! FR-OP-83's `f0_extract` unified API (`(f0_hz, voiced_flag, confidence)`)
//! at landing WP **M5-16** (`docs/milestones.md` §9). The unified
//! `(f0_hz, voiced_flag, confidence)` facade — layered on top of the
//! per-frame `Vec<f32>` these primitives emit and the `F0Frame` rows
//! the neural skeletons emit (`vokra_models::f0::F0Frame`) — is M5-16's
//! job when the M3-17 prosody consumer (or the voice-clone repo) arrives;
//! it is **deliberately out of scope** for this wave (dormancy owner
//! ADR 2026-07-22, memory `[[project-m5-owner-adr-decisions.md]]` item ⑦).
//!
//! Runtime function, NOT an [`vokra_core::OpKind`] variant (same posture as
//! [`crate::resample`] / [`crate::agc`] / [`crate::hpf`] — ADR M4-20 §D-5:
//! per-frame state and whole-signal transforms do not fit the graph-side
//! `OpValue` dispatch surface). No `vokra_core::m5_residual_ops` entry is
//! reserved for the same reason (mirrors `flow_sampler` / `mimi_rvq` /
//! `dac_rvq`).
//!
//! # Red-line: no code lifted from aubio / librosa
//!
//! Both algorithms are **reimplemented from the paper spec**. No code is
//! borrowed from **aubio** (GPL-3.0), **librosa** (ISC), or any other
//! reference implementation — the same posture [`crate::resample`] takes
//! with respect to soxr (LGPL) and rubberband (GPL) (NFR-LC-03/04,
//! CLAUDE.md "GPL/LGPL 回避" red line). Comparisons against reference
//! implementations for numerical checks happen offline, never through
//! vendored code.
//!
//! # Zero-dependency posture (NFR-DS-02)
//!
//! `std` + [`vokra_core::{Result, VokraError}`] only. The root
//! `Cargo.lock` continues to list only `vokra-*` packages. No `libm`, no
//! `serde`, no external crate.

use vokra_core::{Result, VokraError};

pub mod pyin;
pub mod yin;

pub use pyin::{PyinFrame, pyin, pyin_detailed};
pub use yin::yin;

/// Analysis frame size (samples) shared by both extractors. `2048` is the
/// standard YIN default (de Cheveigné & Kawahara 2002 §2, "practical
/// implementations"): large enough that the difference function has
/// meaningful support up to `τ_max ≈ frame_size / 2 ≈ 1024` samples
/// (≈ 15.6 Hz at 16 kHz / ≈ 21.5 Hz at 22.05 kHz), and small enough that
/// the local-stationarity assumption inside the frame is not violated.
pub const DEFAULT_FRAME_SIZE: usize = 2048;

/// Hop between successive analysis frames (samples). `256` samples ≈
/// 11.6 ms at 22.05 kHz / 16 ms at 16 kHz — the standard PyIN /
/// pitch-tracking cadence for downstream prosody consumers.
pub const DEFAULT_HOP: usize = 256;

/// YIN's absolute-threshold dip pick default (paper §3): `0.10`. A CMNDF
/// value below this threshold is treated as a voiced pitch candidate; the
/// dip-picker walks to the local minimum inside the sub-threshold region.
/// PyIN evaluates the same dip-pick at 100 threshold values weighted by
/// [`Beta(2, 18)`](pyin::beta_2_18_pdf), so this default is unused there.
pub const DEFAULT_ABSOLUTE_THRESHOLD: f32 = 0.10;

/// Shared validator for classical F0 arguments.
///
/// Enforces FR-EX-08 loud-fail: `sample_rate` must be non-zero,
/// `fmin` / `fmax` must be positive and finite with `fmin < fmax`, and
/// `fmin * 2` must not exceed `sample_rate` (a violated Nyquist bound
/// would make the difference function's `τ_max = sr / fmin` search
/// interval degenerate).
pub(crate) fn validate_args(sample_rate: u32, fmin: f32, fmax: f32) -> Result<()> {
    if sample_rate == 0 {
        return Err(VokraError::InvalidArgument(
            "f0: sample_rate must be non-zero".to_owned(),
        ));
    }
    if !fmin.is_finite() || !fmax.is_finite() || fmin <= 0.0 || fmax <= 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "f0: fmin and fmax must be positive and finite (fmin={fmin}, fmax={fmax})"
        )));
    }
    if fmin >= fmax {
        return Err(VokraError::InvalidArgument(format!(
            "f0: fmin must be strictly less than fmax (fmin={fmin}, fmax={fmax})"
        )));
    }
    if fmin * 2.0 > sample_rate as f32 {
        return Err(VokraError::InvalidArgument(format!(
            "f0: fmin*2 exceeds sample_rate (Nyquist violated) (fmin={fmin}, sample_rate={sample_rate})"
        )));
    }
    Ok(())
}

/// Compute the `(τ_min, τ_max)` search interval in samples for the given
/// `(sample_rate, fmin, fmax)` triple.
///
/// - `τ_min = floor(sr / fmax)`, clamped to `>= 2` so parabolic
///   interpolation (needs `τ - 1 >= 0`) and the CMNDF sentinel
///   (`d'[0] = 1`, `d'[1]` is often trivially near zero for smooth
///   signals) stay well-defined.
/// - `τ_max = ceil(sr / fmin)`, clamped to `<= DEFAULT_FRAME_SIZE / 2 - 1`
///   so the difference function keeps `>= DEFAULT_FRAME_SIZE / 2` sample
///   pairs (the standard YIN half-frame overlap rule).
///
/// Returns `InvalidArgument` if the clamped interval collapses
/// (`τ_min >= τ_max`) — a signal that the caller asked for a pitch band
/// too narrow for the shared frame size.
pub(crate) fn tau_search_interval(
    sample_rate: u32,
    fmin: f32,
    fmax: f32,
) -> Result<(usize, usize)> {
    let sr = sample_rate as f32;
    let tau_min_raw = (sr / fmax).floor() as usize;
    let tau_max_raw = (sr / fmin).ceil() as usize;
    let tau_min = tau_min_raw.max(2);
    let tau_max = tau_max_raw.min(DEFAULT_FRAME_SIZE / 2 - 1);
    if tau_min >= tau_max {
        return Err(VokraError::InvalidArgument(format!(
            "f0: fmin/fmax yields empty τ search interval after clamping (tau_min={tau_min}, tau_max={tau_max}); widen the range or lower fmin"
        )));
    }
    Ok((tau_min, tau_max))
}

/// Difference function `d[τ] = Σᵢ (x[i] - x[i+τ])²` for
/// `τ ∈ [0, τ_max]` (de Cheveigné & Kawahara 2002 eq. 6). Returns
/// `d` sized `τ_max + 1` with `d[0] = 0` and `d[τ]` non-negative.
///
/// The naïve O(N·τ_max) formulation used here is chosen for correctness /
/// clarity — the FFT-accelerated O(N log N) variant (paper §4) is a future
/// optimisation and not required for the task's per-frame accuracy tests.
pub(crate) fn difference_function(frame: &[f32], tau_max: usize) -> Vec<f32> {
    let mut d = vec![0.0f32; tau_max + 1];
    // d[0] = 0 by construction (Σᵢ (x[i] - x[i])² = 0).
    let n = frame.len();
    for tau in 1..=tau_max {
        if tau >= n {
            // No sample pairs for this lag — leave d[τ] = 0. The CMNDF
            // sentinel below will map this to `d'[τ] = 1` (unvoiced).
            continue;
        }
        let mut sum = 0.0f64;
        let pair_count = n - tau;
        for i in 0..pair_count {
            let diff = f64::from(frame[i]) - f64::from(frame[i + tau]);
            sum += diff * diff;
        }
        d[tau] = sum as f32;
    }
    d
}

/// Cumulative-mean-normalized difference function
/// `d'[τ] = d[τ] · τ / Σⱼ₌₁ᵗ d[j]` with `d'[0] = 1` sentinel
/// (paper §2.4, eq. 8).
///
/// The sentinel `d'[0] = 1` (rather than `d'[0] = 0` from the raw
/// difference function) is what prevents the absolute-threshold dip-picker
/// from spuriously locking onto `τ = 0`. When the running cumulative sum
/// stays zero (a silent frame — every `d[τ] = 0`), the fallback assigns
/// `d'[τ] = 1` so the dip-picker returns `None` → 0 Hz (documented
/// unvoiced convention).
pub(crate) fn cmndf(d: &[f32]) -> Vec<f32> {
    let mut d_prime = vec![0.0f32; d.len()];
    if d.is_empty() {
        return d_prime;
    }
    d_prime[0] = 1.0;
    let mut cumsum = 0.0f64;
    for tau in 1..d.len() {
        cumsum += f64::from(d[tau]);
        if cumsum > 0.0 {
            d_prime[tau] = (f64::from(d[tau]) * (tau as f64) / cumsum) as f32;
        } else {
            // Silent frame — mark as unvoiced.
            d_prime[tau] = 1.0;
        }
    }
    d_prime
}

/// YIN's absolute-threshold dip picker (paper §3): find the first
/// `τ ∈ [τ_min, τ_max]` with `d'[τ] < threshold`, then walk downhill to
/// its local minimum inside the sub-threshold region. Returns `None`
/// (unvoiced) if no CMNDF value in the search interval falls below the
/// threshold.
///
/// Deliberately does **NOT** fall back to a global argmin when the
/// threshold is not crossed — that fallback is a common
/// misinterpretation of the paper and produces spurious pitch on unvoiced
/// frames (silence, breath, fricatives). Callers who want a "best-guess"
/// F0 should raise the threshold instead.
pub(crate) fn absolute_threshold(
    d_prime: &[f32],
    tau_min: usize,
    tau_max: usize,
    threshold: f32,
) -> Option<usize> {
    let mut tau = tau_min;
    while tau <= tau_max {
        if d_prime[tau] < threshold {
            let mut best = tau;
            while best < tau_max && d_prime[best + 1] < d_prime[best] {
                best += 1;
            }
            return Some(best);
        }
        tau += 1;
    }
    None
}

/// Parabolic interpolation on `(τ-1, τ, τ+1)` to refine the CMNDF minimum
/// to sub-sample resolution (paper §5 "refinement"). Returns the refined
/// τ as `f32` — degenerate cases (edge, flat minimum) fall back to the
/// integer τ so the caller never emits a NaN F0.
pub(crate) fn parabolic_interpolation(d_prime: &[f32], tau: usize) -> f32 {
    if tau == 0 || tau + 1 >= d_prime.len() {
        return tau as f32;
    }
    let s0 = f64::from(d_prime[tau - 1]);
    let s1 = f64::from(d_prime[tau]);
    let s2 = f64::from(d_prime[tau + 1]);
    let denom = s0 + s2 - 2.0 * s1;
    if denom.abs() < 1e-10 {
        return tau as f32;
    }
    let offset = 0.5 * (s0 - s2) / denom;
    // Guard: parabolic fit can extrapolate outside (-1, +1) on very flat
    // curves; clamp so the refined τ stays inside the adjacent-sample
    // interval (a well-known YIN implementation robustness note).
    let clamped = offset.clamp(-1.0, 1.0);
    (tau as f64 + clamped) as f32
}

/// Number of analysis frames the shared `(frame_size, hop)` produces
/// for a `pcm_len`-sample input. Returns `0` if `pcm_len < frame_size`
/// — the documented "no partial frames" boundary.
pub(crate) fn num_frames(pcm_len: usize) -> usize {
    if pcm_len < DEFAULT_FRAME_SIZE {
        0
    } else {
        (pcm_len - DEFAULT_FRAME_SIZE) / DEFAULT_HOP + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_args_rejects_zero_sample_rate() {
        let e = validate_args(0, 65.0, 800.0).unwrap_err();
        assert!(matches!(e, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_args_rejects_fmin_ge_fmax() {
        let e = validate_args(16_000, 800.0, 800.0).unwrap_err();
        assert!(matches!(e, VokraError::InvalidArgument(_)));
        let e = validate_args(16_000, 900.0, 800.0).unwrap_err();
        assert!(matches!(e, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_args_rejects_nyquist_violation() {
        // fmin * 2 > sr — asks for the difference function to search below
        // the samplable frequency floor.
        let e = validate_args(1000, 800.0, 900.0).unwrap_err();
        assert!(matches!(e, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_args_accepts_reasonable_range() {
        validate_args(22_050, 65.0, 800.0).unwrap();
        validate_args(16_000, 80.0, 400.0).unwrap();
    }

    #[test]
    fn tau_search_interval_shape() {
        let (tau_min, tau_max) = tau_search_interval(22_050, 65.0, 800.0).unwrap();
        // τ_min = floor(22050 / 800) = 27, τ_max = ceil(22050 / 65) = 340.
        assert_eq!(tau_min, 27);
        assert_eq!(tau_max, 340);
        assert!(tau_min < tau_max);
    }

    #[test]
    fn cmndf_silence_maps_to_unity() {
        // Silent frame — every d[τ] = 0.
        let d = vec![0.0f32; 128];
        let dp = cmndf(&d);
        assert_eq!(dp[0], 1.0);
        for &v in &dp[1..] {
            assert_eq!(v, 1.0, "silence should yield d'[τ] = 1 (unvoiced sentinel)");
        }
    }

    #[test]
    fn num_frames_matches_hop_formula() {
        assert_eq!(num_frames(0), 0);
        assert_eq!(num_frames(DEFAULT_FRAME_SIZE - 1), 0);
        assert_eq!(num_frames(DEFAULT_FRAME_SIZE), 1);
        assert_eq!(num_frames(DEFAULT_FRAME_SIZE + DEFAULT_HOP), 2);
        assert_eq!(num_frames(DEFAULT_FRAME_SIZE + 3 * DEFAULT_HOP), 4);
    }
}
