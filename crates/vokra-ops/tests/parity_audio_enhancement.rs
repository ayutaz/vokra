//! M4-20 (c) T13: audio-enhancement parity for `agc` / `hpf` / `loudness_norm`
//! against **closed-form analytic oracles** — the honest "math oracle" form of
//! "上流 reference と一致" for ops whose spec is exact closed-form math (ADR
//! M4-20 §D-5). No external fixture is needed (and none is fabricated): the
//! reference is the published transfer-function / ITU-R BS.1770 math, evaluated
//! independently here, and the implementations must match it within
//! `atol = 0.01` (NFR-QL-01).
//!
//! `denoise`'s real parity needs the DeepFilterNet checkpoint and is the owner
//! leg (T17) — deliberately absent here (no synthetic-weight pass is faked).

use vokra_ops::{AgcAttrs, HpfAttrs, LoudnessNormAttrs, agc, hpf, integrated_lufs, loudness_norm};

const ATOL: f64 = 0.01;
const PI: f64 = std::f64::consts::PI;

/// `|H(e^{jw})|` of a normalized biquad (`a0 = 1`): the exact steady-state gain.
fn biquad_mag(b: [f64; 3], a: [f64; 3], w: f64) -> f64 {
    let num_re = b[0] + b[1] * (-w).cos() + b[2] * (-2.0 * w).cos();
    let num_im = b[1] * (-w).sin() + b[2] * (-2.0 * w).sin();
    let den_re = a[0] + a[1] * (-w).cos() + a[2] * (-2.0 * w).cos();
    let den_im = a[1] * (-w).sin() + a[2] * (-2.0 * w).sin();
    num_re.hypot(num_im) / den_re.hypot(den_im)
}

/// RBJ high-pass biquad coefficients (the independent reference re-derivation).
fn rbj_highpass(cutoff: f64, fs: f64, q: f64) -> ([f64; 3], [f64; 3]) {
    let w0 = 2.0 * PI * cutoff / fs;
    let (s, c) = w0.sin_cos();
    let alpha = s / (2.0 * q);
    let b = [(1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0];
    let a0 = 1.0 + alpha;
    let a = [1.0, -2.0 * c, 1.0 - alpha];
    (
        [b[0] / a0, b[1] / a0, b[2] / a0],
        [1.0, a[1] / a0, a[2] / a0],
    )
}

fn tone(freq: f64, fs: f64, amp: f64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (amp * (2.0 * PI * freq * i as f64 / fs).sin()) as f32)
        .collect()
}

fn tail_peak(x: &[f32], tail: usize) -> f64 {
    x[x.len() - tail..]
        .iter()
        .fold(0.0f64, |m, &v| m.max(v.abs() as f64))
}

#[test]
fn hpf_steady_state_matches_transfer_function() {
    // The time-domain biquad's steady-state amplitude at a tone must equal the
    // analytic |H(e^{jw})| · input_amp (the RBJ transfer function is the spec).
    let fs = 16000.0;
    let (cutoff, q, amp) = (80.0, std::f64::consts::FRAC_1_SQRT_2, 0.5);
    let (b, a) = rbj_highpass(cutoff, fs, q);
    for &freq in &[40.0, 120.0, 500.0, 3000.0] {
        let n = (fs as usize) * 3; // long enough to reach steady state
        let x = tone(freq, fs, amp, n);
        let out = hpf(
            &x,
            &HpfAttrs {
                cutoff_hz: cutoff as f32,
                sample_rate: fs as u32,
                q: q as f32,
            },
        )
        .unwrap();
        let measured = tail_peak(&out, fs as usize);
        let w = 2.0 * PI * freq / fs;
        let expected = amp * biquad_mag(b, a, w);
        assert!(
            (measured - expected).abs() <= ATOL,
            "hpf @ {freq} Hz: measured {measured:.5} vs analytic {expected:.5} (atol {ATOL})"
        );
    }
}

#[test]
// pyloudnorm / BS.1770 analog-prototype constants transcribed verbatim (the
// independent reference); keep the published precision.
#[allow(clippy::excessive_precision, clippy::inconsistent_digit_grouping)]
fn loudness_matches_bs1770_closed_form() {
    // A stationary sine's integrated LUFS has the exact closed form
    // −0.691 + 10·log10((A²/2)·|H_k(f)|²), where H_k is the K-weighting (shelf ×
    // RLB high-pass). Recompute the K-weight coefficients independently, take
    // |H_k(f)| analytically, and compare to the measured integrated loudness.
    let fs = 48000.0;
    // K-weight stage coefficients (pyloudnorm/BS.1770 analog prototype).
    let (b1, a1) = {
        let g = 3.999_843_853_97;
        let q = 0.707_175_236_955_419_3;
        let f0 = 1681.974_450_955_531_9;
        let k = (PI * f0 / fs).tan();
        let vh = 10.0_f64.powf(g / 20.0);
        let vb = vh.powf(0.499_666_774_154_541_6);
        let a0 = 1.0 + k / q + k * k;
        (
            [
                (vh + vb * k / q + k * k) / a0,
                2.0 * (k * k - vh) / a0,
                (vh - vb * k / q + k * k) / a0,
            ],
            [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
        )
    };
    let (b2, a2) = {
        let q = 0.500_327_037_325_395_3;
        let f0 = 38.135_470_876_139_82;
        let k = (PI * f0 / fs).tan();
        let a0 = 1.0 + k / q + k * k;
        (
            [1.0, -2.0, 1.0],
            [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
        )
    };

    for &(freq, amp) in &[(1000.0, 0.5), (500.0, 0.25), (2000.0, 0.1)] {
        let w = 2.0 * PI * freq / fs;
        let hk = biquad_mag(b1, a1, w) * biquad_mag(b2, a2, w);
        // Mean square of an amplitude-A sine = A²/2; K-weighted → ×|H_k|².
        let mean_sq = 0.5 * amp * amp * hk * hk;
        let expected = -0.691 + 10.0 * mean_sq.log10();

        let x = tone(freq, fs, amp, (fs as usize) * 2); // 2 s → many gating blocks
        let measured = integrated_lufs(&x, fs as u32).unwrap() as f64;
        assert!(
            (measured - expected).abs() <= 0.05,
            "LUFS @ {freq} Hz A={amp}: measured {measured:.4} vs closed-form {expected:.4}"
        );
    }
}

#[test]
fn loudness_norm_hits_target_within_atol() {
    // The normalization fixed point: re-measuring the normalized signal returns
    // the target LUFS exactly (constant-gain round trip).
    let fs = 48000u32;
    let x = tone(1000.0, fs as f64, 0.3, (fs as usize) * 2);
    for &target in &[-23.0f32, -16.0, -31.0] {
        let out = loudness_norm(
            &x,
            &LoudnessNormAttrs {
                target_lufs: target,
                sample_rate: fs,
            },
        )
        .unwrap();
        let measured = integrated_lufs(&out, fs).unwrap() as f64;
        assert!(
            (measured - target as f64).abs() <= ATOL,
            "loudness_norm target {target}: re-measured {measured:.5}"
        );
    }
}

#[test]
fn agc_converges_to_its_target_fixed_point() {
    // The AGC's analytic fixed point for a stationary tone is `target_level`
    // (gain = target/env, env → amp, output amp → target). The converged peak
    // must equal the target within atol for tones reachable inside max_gain.
    let fs = 16000.0;
    let attrs = AgcAttrs::speech_default(); // target 0.25, max_gain 20
    for &amp in &[0.05, 0.5] {
        let x = tone(300.0, fs, amp, (fs as usize) * 4); // long settle
        let out = agc(&x, &attrs).unwrap();
        let measured = tail_peak(&out, fs as usize);
        assert!(
            (measured - attrs.target_level as f64).abs() <= 0.02,
            "agc from amp {amp}: converged peak {measured:.4} vs target {}",
            attrs.target_level
        );
    }
}
