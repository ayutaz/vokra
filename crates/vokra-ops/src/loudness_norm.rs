//! `loudness_norm` — loudness normalization (M4-20 (c), FR-OP-63): measures the
//! integrated loudness in LUFS per **ITU-R BS.1770 / EBU R128** and scales the
//! signal to a target LUFS (CLAUDE.md「loudness_norm — LUFS/EBU R128 対応」).
//!
//! # Runtime function, not an `OpKind` variant (ADR M4-20 §D-5)
//!
//! A whole-signal transform exposed as a first-class API function, not an
//! `OpKind` variant. `LoudnessNormAttrs` is defined here.
//!
//! # Spec (published, not invented)
//!
//! - **K-weighting**: the two-stage biquad of ITU-R BS.1770 — a high-shelf
//!   "pre-filter" (head) followed by an RLB high-pass. Coefficients are derived
//!   from the analog prototype via the bilinear transform at the signal's
//!   sample rate (the pyloudnorm method, BSD-3; the analog-prototype constants
//!   — shelf `f0 ≈ 1681.97 Hz`, `G ≈ 4 dB`, `Q ≈ 0.7071`; high-pass
//!   `f0 ≈ 38.135 Hz`, `Q ≈ 0.5003` — reproduce the BS.1770 48 kHz reference
//!   coefficients and are transcribed, not invented).
//! - **Integrated loudness**: `−0.691 + 10·log10(mean_square)` over 400 ms
//!   blocks (100 ms hop, 75 % overlap) with the BS.1770-4 two-stage gating
//!   (absolute gate −70 LUFS, relative gate −10 LU). Signals shorter than one
//!   block fall back to an ungated whole-signal measure.
//!
//! # Mathematical oracles (T13)
//!
//! The measurement obeys exact scaling: doubling the amplitude raises the LUFS
//! by exactly `20·log10(2) ≈ 6.0206 dB` (a linear filter + `10·log10` of the
//! mean square), and normalizing to a target then re-measuring returns the
//! target (constant-gain round trip). Both hold regardless of the exact
//! K-weighting coefficients, so the parity test asserts them directly.

use vokra_core::{Result, VokraError};

/// The BS.1770 loudness offset (`−0.691 dB`, the mono channel-weighting term).
const LOUDNESS_OFFSET: f64 = -0.691;

/// Attributes for [`loudness_norm`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoudnessNormAttrs {
    /// Target integrated loudness in LUFS (EBU R128 default `−23.0`; streaming
    /// platforms commonly use `−14.0`).
    pub target_lufs: f32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl LoudnessNormAttrs {
    /// EBU R128 default target (`−23 LUFS`).
    pub fn ebu_r128(sample_rate: u32) -> Self {
        Self {
            target_lufs: -23.0,
            sample_rate,
        }
    }
}

/// Direct-Form-I biquad over `x` (f64), coefficients already normalized so
/// `a0 = 1` (`a = [a1, a2]`, `b = [b0, b1, b2]`).
fn biquad(x: &[f64], b: [f64; 3], a: [f64; 2]) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
    for &x0 in x {
        let y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 - a[0] * y1 - a[1] * y2;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
        out.push(y0);
    }
    out
}

/// Applies the BS.1770 K-weighting (shelf → RLB high-pass) to `signal`.
// The analog-prototype constants are transcribed verbatim from pyloudnorm /
// ITU-R BS.1770 (documented reference values); keep the published precision.
#[allow(clippy::excessive_precision, clippy::inconsistent_digit_grouping)]
fn k_weight(signal: &[f32], fs: u32) -> Vec<f64> {
    let rate = fs as f64;
    // Stage 1 — high-shelf pre-filter (pyloudnorm/BS.1770 analog prototype).
    let (b1s, a1s) = {
        let g = 3.999_843_853_97_f64;
        let q = 0.707_175_236_955_419_3_f64;
        let f0 = 1681.974_450_955_531_9_f64;
        let k = (std::f64::consts::PI * f0 / rate).tan();
        let vh = 10.0_f64.powf(g / 20.0);
        let vb = vh.powf(0.499_666_774_154_541_6);
        let a0 = 1.0 + k / q + k * k;
        (
            [
                (vh + vb * k / q + k * k) / a0,
                2.0 * (k * k - vh) / a0,
                (vh - vb * k / q + k * k) / a0,
            ],
            [2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
        )
    };
    // Stage 2 — RLB high-pass.
    let (b2s, a2s) = {
        let q = 0.500_327_037_325_395_3_f64;
        let f0 = 38.135_470_876_139_82_f64;
        let k = (std::f64::consts::PI * f0 / rate).tan();
        let a0 = 1.0 + k / q + k * k;
        (
            [1.0, -2.0, 1.0],
            [2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
        )
    };
    let x: Vec<f64> = signal.iter().map(|&s| s as f64).collect();
    let s1 = biquad(&x, b1s, a1s);
    biquad(&s1, b2s, a2s)
}

/// `−0.691 + 10·log10(mean_square)`, or `−∞` for a non-positive mean square.
fn loudness_of(mean_square: f64) -> f64 {
    if mean_square > 0.0 {
        LOUDNESS_OFFSET + 10.0 * mean_square.log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// Integrated loudness (LUFS) of a mono signal per ITU-R BS.1770-4 (gated), or
/// `f64::NEG_INFINITY` for silence. Public for callers / the parity oracle.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] for a zero sample rate or a non-finite
/// sample.
pub fn integrated_lufs(signal: &[f32], sample_rate: u32) -> Result<f32> {
    if sample_rate == 0 {
        return Err(VokraError::InvalidArgument(
            "loudness: sample_rate is 0".into(),
        ));
    }
    if signal.iter().any(|s| !s.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "loudness: input has a non-finite sample".into(),
        ));
    }
    Ok(measure(signal, sample_rate) as f32)
}

fn measure(signal: &[f32], fs: u32) -> f64 {
    if signal.is_empty() {
        return f64::NEG_INFINITY;
    }
    let y = k_weight(signal, fs);
    let block = (0.4 * fs as f64).round() as usize;
    let hop = (0.1 * fs as f64).round() as usize;

    // Short-signal fallback: fewer than one full 400 ms block → ungated
    // whole-signal mean square (the gating machinery needs blocks).
    if block == 0 || hop == 0 || y.len() < block {
        let ms = y.iter().map(|v| v * v).sum::<f64>() / y.len() as f64;
        return loudness_of(ms);
    }

    // Per-block mean squares (400 ms / 100 ms hop).
    let mut zs = Vec::new();
    let mut start = 0;
    while start + block <= y.len() {
        let ms = y[start..start + block].iter().map(|v| v * v).sum::<f64>() / block as f64;
        zs.push(ms);
        start += hop;
    }

    // Absolute gate (−70 LUFS).
    let abs_gated: Vec<f64> = zs
        .iter()
        .copied()
        .filter(|&z| loudness_of(z) >= -70.0)
        .collect();
    if abs_gated.is_empty() {
        return f64::NEG_INFINITY;
    }
    // Relative threshold = loudness of the mean of the abs-gated blocks − 10 LU.
    let mean_abs = abs_gated.iter().sum::<f64>() / abs_gated.len() as f64;
    let gamma_r = loudness_of(mean_abs) - 10.0;
    // Relative gate.
    let rel_gated: Vec<f64> = zs
        .iter()
        .copied()
        .filter(|&z| loudness_of(z) >= gamma_r)
        .collect();
    let gated = if rel_gated.is_empty() {
        &abs_gated
    } else {
        &rel_gated
    };
    let mean_rel = gated.iter().sum::<f64>() / gated.len() as f64;
    loudness_of(mean_rel)
}

/// Normalizes `input` to `attrs.target_lufs`, returning the gain-scaled signal
/// (same length). A silent input (measured `−∞`) is returned unchanged (there
/// is no defined gain to a target); this is documented, not a silent error.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] for a zero sample rate or a non-finite input
/// sample (FR-EX-08).
pub fn loudness_norm(input: &[f32], attrs: &LoudnessNormAttrs) -> Result<Vec<f32>> {
    let measured = integrated_lufs(input, attrs.sample_rate)? as f64;
    if !measured.is_finite() {
        // Silence: no defined normalization gain — return unchanged.
        return Ok(input.to_vec());
    }
    // Gain in dB = target − measured; linear = 10^(dB/20).
    let gain = 10.0_f64.powf((attrs.target_lufs as f64 - measured) / 20.0) as f32;
    Ok(input.iter().map(|&s| s * gain).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f32, fs: u32, amp: f32, secs: f32) -> Vec<f32> {
        let n = (fs as f32 * secs) as usize;
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / fs as f32).sin())
            .collect()
    }

    #[test]
    fn doubling_amplitude_raises_lufs_by_6db() {
        // Exact BS.1770 scaling: 2× amplitude → +20·log10(2) ≈ 6.0206 LUFS.
        let fs = 48000;
        let a = tone(1000.0, fs, 0.25, 1.0);
        let b: Vec<f32> = a.iter().map(|v| v * 2.0).collect();
        let la = integrated_lufs(&a, fs).unwrap();
        let lb = integrated_lufs(&b, fs).unwrap();
        let delta = lb - la;
        assert!(
            (delta - 6.0206).abs() < 0.01,
            "doubling must add ~6.02 LUFS, got {delta}"
        );
    }

    #[test]
    fn normalize_round_trip_hits_target() {
        // Normalizing to a target then re-measuring returns the target
        // (constant-gain round trip), for several targets.
        let fs = 48000;
        let x = tone(1000.0, fs, 0.1, 1.5);
        for target in [-23.0f32, -14.0, -30.0] {
            let out = loudness_norm(
                &x,
                &LoudnessNormAttrs {
                    target_lufs: target,
                    sample_rate: fs,
                },
            )
            .unwrap();
            let remeasured = integrated_lufs(&out, fs).unwrap();
            assert!(
                (remeasured - target).abs() < 0.05,
                "re-measured {remeasured} must match target {target}"
            );
        }
    }

    #[test]
    fn silence_is_returned_unchanged() {
        let fs = 48000;
        let x = vec![0.0f32; fs as usize];
        let out = loudness_norm(&x, &LoudnessNormAttrs::ebu_r128(fs)).unwrap();
        assert_eq!(out, x, "silence has no defined gain; returned unchanged");
        assert!(!integrated_lufs(&x, fs).unwrap().is_finite());
    }

    #[test]
    fn short_signal_uses_ungated_fallback_and_still_scales() {
        // A 50 ms signal (< one 400 ms block) still measures and obeys the 6 dB
        // scaling via the ungated fallback path.
        let fs = 48000;
        let a = tone(1000.0, fs, 0.2, 0.05);
        let b: Vec<f32> = a.iter().map(|v| v * 2.0).collect();
        let delta = integrated_lufs(&b, fs).unwrap() - integrated_lufs(&a, fs).unwrap();
        assert!(
            (delta - 6.0206).abs() < 0.02,
            "short-signal scaling {delta}"
        );
    }

    #[test]
    fn rejects_zero_rate_and_nonfinite() {
        assert!(integrated_lufs(&[0.1], 0).is_err());
        assert!(integrated_lufs(&[f32::NAN], 48000).is_err());
        assert!(loudness_norm(&[f32::NAN], &LoudnessNormAttrs::ebu_r128(48000)).is_err());
    }
}
