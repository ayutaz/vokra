//! Arbitrary-rate audio resampling (M1-06a; FR-OP-64 `resample`).
//!
//! Converts a real PCM stream from `in_rate` to `out_rate` with a
//! **Kaiser-windowed-sinc interpolation kernel**. This is the standard
//! band-limited-interpolation / polyphase resampler of textbook DSP
//! (Oppenheim & Schafer §4.6; Kaiser 1974) — a *from-scratch re-derivation*,
//! not a port. It copies no code from **soxr** (LGPL) or **rubberband** (GPL);
//! those are neither vendored nor referenced (NFR-LC-03/04, CLAUDE.md red
//! line). There is no external crate: the [`sinc`](crate::window) and Kaiser
//! window (`I0` Bessel series) are first-party in [`crate::window`].
//!
//! # Algorithm
//!
//! Output sample `j` sits at input-sample position `c = j · in_rate/out_rate`.
//! Its value is a weighted sum of the input samples within a finite support
//! around `c`:
//!
//! ```text
//!   y[j] = ( Σ_i x[i] · h(c − i) ) / ( Σ_i h(c − i) )
//!   h(τ) = cutoff · sinc(cutoff · τ) · kaiser(τ / R)      for |τ| < R,  else 0
//! ```
//!
//! - `cutoff = min(1, out_rate/in_rate)` places the low-pass corner at the
//!   lower of the two Nyquist frequencies — an anti-imaging filter when
//!   upsampling, an anti-aliasing filter when downsampling.
//! - `R = half / cutoff` is the support half-width in input samples, where
//!   `half` is the number of sinc zero-crossings retained per side (a quality
//!   knob).
//! - The **per-output normalization** (dividing by the tap sum) pins the DC /
//!   passband gain to exactly `1.0` for every fractional phase, so a constant
//!   input maps to itself and a linear ramp is preserved on the interior.
//!
//! Equal rates short-circuit to a bit-exact copy. This module ships the
//! **one-shot** path; the streaming wrapper (phase-accumulator + input-history
//! carry-over across chunks) is M1-08, and [`SincResampler`] is laid out with
//! that split already in mind.

use vokra_core::{Result, VokraError};

use crate::window::{bessel_i0, sinc};

/// The default resampling quality used by the frontend chain
/// ([`crate::preprocess::apply_frontend`]).
///
/// `frontend_spec` stores no resampler quality, so the chain fixes a strong
/// default here (see [`quality_params`]).
pub const DEFAULT_QUALITY: u8 = 5;

/// Resamples `input` from `in_rate` Hz to `out_rate` Hz.
///
/// `quality` selects the filter's zero-crossing count and Kaiser β (see
/// [`quality_params`]); higher is sharper and slower. Equal rates return a
/// bit-exact copy of `input`. The output length is
/// `round(input.len() · out_rate / in_rate)`.
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`] if `in_rate` or `out_rate` is zero.
///
/// # Examples
///
/// ```
/// let x = vec![0.0f32, 1.0, 0.0, -1.0];
/// // 1:1 resampling is the exact identity.
/// let y = vokra_ops::resample(&x, 16_000, 16_000, 5).unwrap();
/// assert_eq!(x, y);
/// ```
pub fn resample(input: &[f32], in_rate: u32, out_rate: u32, quality: u8) -> Result<Vec<f32>> {
    if in_rate == 0 || out_rate == 0 {
        return Err(VokraError::InvalidArgument(
            "resample: in_rate and out_rate must be non-zero".to_owned(),
        ));
    }
    if in_rate == out_rate {
        return Ok(input.to_vec());
    }
    Ok(SincResampler::new(in_rate, out_rate, quality).resample(input))
}

/// Maps a `quality` byte to `(half, beta)`: the sinc half-length in
/// zero-crossings and the Kaiser shape parameter β.
///
/// This table is **Vokra's own** internally-consistent design (we reimplement,
/// so it need not match speexdsp/soxr): larger β lowers the stopband floor,
/// larger `half` narrows the transition band. The realized passband ripple and
/// stopband attenuation are what the resampler tests assert against, rather
/// than any borrowed magic numbers. Values `>= 10` saturate to the top row.
pub fn quality_params(quality: u8) -> (usize, f64) {
    match quality {
        0 => (4, 6.0),
        1 => (6, 6.5),
        2 => (8, 7.0),
        3 => (10, 8.0),
        4 => (12, 8.5),
        5 => (16, 9.0),
        6 => (20, 10.0),
        7 => (24, 11.0),
        8 => (32, 12.0),
        9 => (48, 13.0),
        _ => (64, 14.0),
    }
}

/// A configured windowed-sinc resampler.
///
/// Holds the precomputed kernel geometry for one `(in_rate, out_rate, quality)`
/// triple. M1-06 exposes only the one-shot [`resample`](Self::resample); the
/// M1-08 streaming wrapper will extend this with a fractional phase accumulator
/// and an input-history ring buffer so successive chunks join seamlessly —
/// hence the config is split out from the transient state here.
struct SincResampler {
    /// Output-to-input rate ratio (`out_rate / in_rate`).
    ratio: f64,
    /// Input samples advanced per output sample (`in_rate / out_rate`).
    step: f64,
    /// Low-pass cutoff as a fraction of the input Nyquist, `min(1, ratio)`.
    cutoff: f64,
    /// Kernel support half-width in input samples, `half / cutoff`.
    radius: f64,
    /// Kaiser shape parameter β.
    beta: f64,
    /// `I0(β)`, precomputed to normalize the Kaiser envelope.
    i0_beta: f64,
}

impl SincResampler {
    fn new(in_rate: u32, out_rate: u32, quality: u8) -> Self {
        let (half, beta) = quality_params(quality);
        let ratio = f64::from(out_rate) / f64::from(in_rate);
        let cutoff = ratio.min(1.0);
        Self {
            ratio,
            step: f64::from(in_rate) / f64::from(out_rate),
            cutoff,
            radius: half as f64 / cutoff,
            beta,
            i0_beta: bessel_i0(beta),
        }
    }

    /// The interpolation kernel `h(τ)` for an input-sample offset `τ`.
    ///
    /// Zero outside the open support `(−radius, radius)`; the endpoints are
    /// excluded symmetrically so the retained tap set stays symmetric about the
    /// output position (which is what preserves DC and linear signals).
    fn kernel(&self, tau: f64) -> f64 {
        let r = tau / self.radius;
        if r.abs() >= 1.0 {
            return 0.0;
        }
        let env = bessel_i0(self.beta * (1.0 - r * r).sqrt()) / self.i0_beta;
        self.cutoff * sinc(self.cutoff * tau) * env
    }

    fn resample(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len();
        if n == 0 {
            return Vec::new();
        }
        let out_len = (n as f64 * self.ratio).round() as usize;
        let mut out = Vec::with_capacity(out_len);
        for j in 0..out_len {
            let center = j as f64 * self.step;
            let lo = (center - self.radius).ceil() as isize;
            let hi = (center + self.radius).floor() as isize;
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for i in lo..=hi {
                let h = self.kernel(center - i as f64);
                den += h;
                if i >= 0 && (i as usize) < n {
                    num += f64::from(input[i as usize]) * h;
                }
            }
            let y = if den != 0.0 { num / den } else { 0.0 };
            out.push(y as f32);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stft::stft;
    use vokra_core::ir::graph::StftAttrs;

    const TAU: f64 = std::f64::consts::TAU;

    fn sine(freq: f64, rate: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|t| (TAU * freq * t as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    fn rms(x: &[f32]) -> f64 {
        if x.is_empty() {
            return 0.0;
        }
        (x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64).sqrt()
    }

    /// Dominant STFT bin of an interior frame, in Hz.
    fn dominant_freq(signal: &[f32], rate: u32, n_fft: usize) -> f64 {
        let attrs = StftAttrs::new(n_fft, n_fft / 4);
        let spec = stft(signal, &attrs).unwrap();
        let f = spec.frames / 2;
        let base = f * spec.bins;
        let (argmax, _) = (0..spec.bins)
            .map(|b| spec.re[base + b].hypot(spec.im[base + b]))
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        argmax as f64 * f64::from(rate) / n_fft as f64
    }

    #[test]
    fn equal_rate_is_bit_exact_identity() {
        let x: Vec<f32> = (0..500).map(|i| (i as f32 * 0.013).sin()).collect();
        let y = resample(&x, 44_100, 44_100, 5).unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn zero_rate_is_rejected() {
        assert!(matches!(
            resample(&[0.0; 4], 0, 16_000, 5),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            resample(&[0.0; 4], 16_000, 0, 5),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn empty_input_gives_empty_output() {
        assert!(resample(&[], 16_000, 8_000, 5).unwrap().is_empty());
    }

    #[test]
    fn output_length_follows_the_rounding_formula() {
        // Our defined length contract: round(in_len * out/in).
        let cases = [
            (1000usize, 16_000u32, 8_000u32, 500usize),
            (1000, 8_000, 16_000, 2000),
            (1000, 16_000, 24_000, 1500),
            (999, 3, 2, 666),            // round(999*2/3) = 666
            (1000, 44_100, 16_000, 363), // round(1000*16000/44100) = 363
        ];
        for (n, fin, fout, want) in cases {
            let y = resample(&vec![0.0f32; n], fin, fout, 5).unwrap();
            assert_eq!(y.len(), want, "{n} @ {fin}->{fout}");
        }
    }

    #[test]
    fn determinism_bit_identical() {
        let x = sine(220.0, 16_000, 4096);
        let a = resample(&x, 16_000, 22_050, 6).unwrap();
        let b = resample(&x, 16_000, 22_050, 6).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn constant_passband_gain_is_unity() {
        // A DC input reproduces itself on the interior (per-phase tap-sum
        // normalization pins the passband gain to 1.0).
        let x = vec![0.75f32; 4000];
        for &(fin, fout) in &[(16_000u32, 8_000u32), (8_000, 16_000), (16_000, 24_000)] {
            let y = resample(&x, fin, fout, 5).unwrap();
            let lo = y.len() / 4;
            let hi = 3 * y.len() / 4;
            for &v in &y[lo..hi] {
                assert!((v - 0.75).abs() < 1e-4, "{fin}->{fout}: {v}");
            }
        }
    }

    #[test]
    fn linear_ramp_is_preserved_on_the_interior() {
        // For integer up/down ratios the retained tap set is symmetric about
        // the output position, so a linear signal maps exactly (to f32) to the
        // ramp sampled at the new rate — a clean analytic oracle. Cases chosen
        // so every output phase is integer or half-integer (symmetric taps):
        // 1:2 up, 2:1 down, 3:1 down.
        let n = 600;
        let x: Vec<f32> = (0..n).map(|i| 0.01 * i as f32).collect();
        for &(fin, fout) in &[(1u32, 2u32), (2, 1), (3, 1)] {
            let y = resample(&x, fin, fout, 5).unwrap();
            let step = f64::from(fin) / f64::from(fout);
            let radius_in = 16.0 / f64::from(fout.min(fin)) * f64::from(fin); // half/cutoff
            let guard = (radius_in / step).ceil() as usize + 2;
            let end = y.len().saturating_sub(guard);
            for (j, &yj) in y.iter().enumerate().take(end).skip(guard) {
                let want = 0.01_f64 * (j as f64 * step);
                assert!(
                    (f64::from(yj) - want).abs() < 2e-3,
                    "{fin}->{fout} j={j}: {yj} vs {want}"
                );
            }
        }
    }

    #[test]
    fn upsample_preserves_a_tone_frequency() {
        // 1 kHz sine, 16 kHz -> 48 kHz: the dominant bin stays at ~1 kHz.
        let x = sine(1000.0, 16_000, 4000);
        let y = resample(&x, 16_000, 48_000, 6).unwrap();
        let f = dominant_freq(&y, 48_000, 2048);
        assert!((f - 1000.0).abs() < 48_000.0 / 2048.0, "dominant {f} Hz");
    }

    #[test]
    fn downsample_passband_tone_survives_with_unit_gain() {
        // 500 Hz is well below the 1 kHz post-downsample Nyquist: it passes at
        // ~unit amplitude and keeps its frequency.
        let x = sine(500.0, 8_000, 8000);
        let y = resample(&x, 8_000, 2_000, 6).unwrap();
        let interior = &y[200..y.len() - 200];
        let ratio = rms(interior) / rms(&x);
        assert!((ratio - 1.0).abs() < 0.05, "passband gain {ratio}");
        let f = dominant_freq(interior, 2_000, 512);
        assert!((f - 500.0).abs() < 2_000.0 / 512.0, "dominant {f} Hz");
    }

    #[test]
    fn downsample_rejects_out_of_band_tone() {
        // A 3 kHz tone is far above the 1 kHz post-downsample Nyquist and deep
        // in the designed stopband: it is attenuated by >30 dB (the anti-alias
        // oracle the Kaiser-sinc filter is built for).
        let x = sine(3000.0, 8_000, 8000);
        let y = resample(&x, 8_000, 2_000, 6).unwrap();
        let interior = &y[200..y.len() - 200];
        let atten_db = 20.0 * (rms(&x) / rms(interior).max(1e-12)).log10();
        assert!(atten_db > 30.0, "only {atten_db} dB of attenuation");
    }

    #[test]
    fn up_then_down_roundtrip_matches_interior() {
        // A band-limited signal survives R -> 2R -> R and R -> 3R/2 -> R on the
        // interior (edges carry the filter transient and are excluded).
        let n = 4000;
        let x: Vec<f32> = (0..n)
            .map(|t| {
                let s = TAU * t as f64 / 8_000.0;
                (0.6 * (300.0 * s).sin() + 0.3 * (700.0 * s).sin()) as f32
            })
            .collect();
        for &mid in &[16_000u32, 12_000u32] {
            let up = resample(&x, 8_000, mid, 7).unwrap();
            let back = resample(&up, mid, 8_000, 7).unwrap();
            assert_eq!(back.len(), n);
            let mut max = 0.0f32;
            for i in 300..n - 300 {
                max = max.max((x[i] - back[i]).abs());
            }
            assert!(max < 2e-2, "roundtrip via {mid}: max err {max}");
        }
    }

    #[test]
    fn quality_table_is_monotonic_and_saturates() {
        let mut prev = (0usize, 0.0f64);
        for q in 0u8..=9 {
            let (half, beta) = quality_params(q);
            assert!(half >= prev.0 && beta >= prev.1, "q={q} not monotonic");
            prev = (half, beta);
        }
        // Values past the table saturate to the top row.
        assert_eq!(quality_params(10), quality_params(255));
    }
}
