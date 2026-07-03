//! Amplitude preprocessing ops and the `frontend_spec` chain (M1-06b; FR-OP-64).
//!
//! Two one-pass time-domain filters that sit ahead of framing / STFT, plus the
//! [`apply_frontend`] chain that drives them from a
//! [`FrontendSpec`](vokra_core::FrontendSpec):
//!
//! - [`dc_offset_remove`] — subtracts the per-utterance mean (`vokra.frontend.
//!   dc_offset_removal`). Removing DC before framing keeps the STFT bin-0 from
//!   being dominated by a constant offset.
//! - [`pre_emphasis`] — the first-order high-pass `y[n] = x[n] − a·x[n−1]`
//!   (`vokra.frontend.pre_emphasis`), with the Kaldi boundary `y[0] = (1−a)·x[0]`.
//!
//! # Bit-exactness (reviewer-C note #2)
//!
//! `frontend_spec` stores only a *bool* for DC removal and a single coefficient
//! for pre-emphasis, so the exact variant is pinned to the first consuming
//! model's documented front-end (Whisper, the currently shipped model, sets
//! `dc_offset_removal = false` and `pre_emphasis = 0.0`, so these are
//! forward-looking ops). The chosen conventions — per-utterance mean and the
//! Kaldi `y[0]` boundary — are documented on each function; alternatives
//! (per-frame mean / leaky DC blocker; `y[0] = x[0]`) are flagged for that
//! decision.

use vokra_core::{FrontendSpec, Result};

use crate::resample::{DEFAULT_QUALITY, resample};

/// Removes the DC offset by subtracting the **per-utterance mean**.
///
/// Deterministic and gain-preserving (`y = x − mean(x)`); an empty input yields
/// an empty output. The mean is accumulated in `f64` to avoid catastrophic
/// cancellation on long inputs, then applied in `f32`.
///
/// # Examples
///
/// ```
/// // A constant signal is pure DC: removing it yields silence.
/// let y = vokra_ops::dc_offset_remove(&[0.5f32; 8]);
/// assert!(y.iter().all(|&v| v == 0.0));
/// ```
pub fn dc_offset_remove(pcm: &[f32]) -> Vec<f32> {
    if pcm.is_empty() {
        return Vec::new();
    }
    let mean = (pcm.iter().map(|&x| f64::from(x)).sum::<f64>() / pcm.len() as f64) as f32;
    pcm.iter().map(|&x| x - mean).collect()
}

/// Applies first-order pre-emphasis `y[n] = x[n] − coeff·x[n−1]`.
///
/// The boundary uses the Kaldi convention `y[0] = (1 − coeff)·x[0]` (as if
/// `x[−1] = x[0]`). `coeff = 0.0` is the exact identity; an empty input yields
/// an empty output. It is exactly invertible by de-emphasis
/// (`x[n] = y[n] + coeff·x[n−1]`, `x[0] = y[0] / (1 − coeff)`).
///
/// # Examples
///
/// ```
/// // coeff = 0 is the identity filter.
/// let x = vec![1.0f32, 2.0, 3.0];
/// assert_eq!(vokra_ops::pre_emphasis(&x, 0.0), x);
/// ```
pub fn pre_emphasis(pcm: &[f32], coeff: f32) -> Vec<f32> {
    if pcm.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(pcm.len());
    out.push((1.0 - coeff) * pcm[0]);
    for w in pcm.windows(2) {
        out.push(w[1] - coeff * w[0]);
    }
    out
}

/// Runs the `frontend_spec` amplitude chain over raw PCM captured at `in_rate`.
///
/// In order: **resample** `in_rate → spec.sample_rate` (skipped, as a bit-exact
/// copy, when the rates already agree; uses [`DEFAULT_QUALITY`]) → **DC
/// removal** (only if `spec.dc_offset_removal`) → **pre-emphasis** (only if
/// `spec.pre_emphasis != 0.0`). The result is PCM at `spec.sample_rate`, ready
/// for the STFT / log-mel front-end.
///
/// This ordering — resample first, then the amplitude filters, then
/// framing/STFT downstream — must match the target model's documented
/// front-end; it is flagged for confirmation when a model that enables these
/// stages lands.
///
/// # Errors
///
/// Propagates [`VokraError::InvalidArgument`](vokra_core::VokraError) from the
/// resample stage (e.g. a zero `spec.sample_rate` when a rate change is needed).
pub fn apply_frontend(pcm: &[f32], in_rate: u32, spec: &FrontendSpec) -> Result<Vec<f32>> {
    let mut x = if in_rate == spec.sample_rate {
        pcm.to_vec()
    } else {
        resample(pcm, in_rate, spec.sample_rate, DEFAULT_QUALITY)?
    };
    if spec.dc_offset_removal {
        x = dc_offset_remove(&x);
    }
    if spec.pre_emphasis != 0.0 {
        x = pre_emphasis(&x, spec.pre_emphasis);
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn whisper_like_spec(sample_rate: u32) -> FrontendSpec {
        FrontendSpec {
            n_fft: 400,
            hop: 160,
            win_length: 400,
            window_type: "hann".to_owned(),
            mel_norm: "slaney".to_owned(),
            htk_mode: false,
            fmin: 0.0,
            fmax: 8000.0,
            n_mels: 80,
            pad_mode: "reflect".to_owned(),
            dc_offset_removal: false,
            pre_emphasis: 0.0,
            sample_rate,
        }
    }

    #[test]
    fn dc_removal_zeros_a_constant_and_centers_the_mean() {
        assert!(dc_offset_remove(&[3.0f32; 16]).iter().all(|&v| v == 0.0));
        assert!(dc_offset_remove(&[]).is_empty());

        let x: Vec<f32> = (0..1000).map(|i| 1.25 + (i as f32 * 0.01).sin()).collect();
        let y = dc_offset_remove(&x);
        let mean_y = y.iter().map(|&v| f64::from(v)).sum::<f64>() / y.len() as f64;
        assert!(mean_y.abs() < 1e-4, "residual mean {mean_y}");
    }

    #[test]
    fn dc_removal_leaves_zero_mean_input_essentially_unchanged() {
        // A whole number of sine periods has (numerically) zero mean:
        // sum_{i<N} sin(2*pi*k*i/N) = 0 for integer k, so removal is a near
        // no-op. (A non-integer number of periods carries a real DC term and
        // would legitimately change.)
        let n = 512;
        let x: Vec<f32> = (0..n)
            .map(|i| (std::f64::consts::TAU * 4.0 * i as f64 / n as f64).sin() as f32)
            .collect();
        let y = dc_offset_remove(&x);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn pre_emphasis_identity_and_closed_forms() {
        let x: Vec<f32> = (0..64).map(|i| 0.3 + 0.02 * i as f32).collect();
        // coeff = 0 is the exact identity.
        assert_eq!(pre_emphasis(&x, 0.0), x);
        assert!(pre_emphasis(&[], 0.5).is_empty());

        // Constant c => y[0] = (1-a)c and y[n>=1] = c - a c = (1-a)c.
        let a = 0.97f32;
        let c = 0.5f32;
        let y = pre_emphasis(&[c; 32], a);
        for &v in &y {
            assert!((v - (1.0 - a) * c).abs() < 1e-6, "{v}");
        }
    }

    #[test]
    fn pre_emphasis_is_invertible_by_de_emphasis() {
        let a = 0.97f32;
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.07).sin() * 0.8).collect();
        let y = pre_emphasis(&x, a);
        // De-emphasis: x[0] = y[0]/(1-a); x[n] = y[n] + a*x[n-1].
        let mut rec = Vec::with_capacity(x.len());
        rec.push(y[0] / (1.0 - a));
        for n in 1..y.len() {
            rec.push(y[n] + a * rec[n - 1]);
        }
        for (a, b) in x.iter().zip(&rec) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn all_off_spec_at_matching_rate_is_identity() {
        let x: Vec<f32> = (0..300).map(|i| (i as f32 * 0.02).cos()).collect();
        let spec = whisper_like_spec(16_000);
        let y = apply_frontend(&x, 16_000, &spec).unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn chain_applies_dc_then_preemph_in_order() {
        let x: Vec<f32> = (0..300).map(|i| 0.4 + (i as f32 * 0.03).sin()).collect();
        let mut spec = whisper_like_spec(16_000);
        spec.dc_offset_removal = true;
        spec.pre_emphasis = 0.97;
        // Same rate: resample is skipped, so the chain == preemph(dc(x)).
        let got = apply_frontend(&x, 16_000, &spec).unwrap();
        let want = pre_emphasis(&dc_offset_remove(&x), 0.97);
        assert_eq!(got, want);
    }

    #[test]
    fn chain_resamples_first_when_rates_differ() {
        let x: Vec<f32> = (0..400).map(|i| (i as f32 * 0.02).sin()).collect();
        let mut spec = whisper_like_spec(8_000);
        spec.dc_offset_removal = true;
        spec.pre_emphasis = 0.5;
        let got = apply_frontend(&x, 16_000, &spec).unwrap();
        // Reference: resample -> dc -> preemph, in that order.
        let r = resample(&x, 16_000, 8_000, DEFAULT_QUALITY).unwrap();
        let want = pre_emphasis(&dc_offset_remove(&r), 0.5);
        assert_eq!(got, want);
        assert_eq!(got.len(), r.len());
    }

    #[test]
    fn chain_surfaces_resample_errors() {
        // A zero target rate with a rate change needed is an error, not silence.
        let spec = whisper_like_spec(0);
        assert!(apply_frontend(&[0.0f32; 8], 16_000, &spec).is_err());
    }
}
