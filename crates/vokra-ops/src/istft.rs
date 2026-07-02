//! Inverse STFT via weighted overlap-add (M0-04-T12; FR-OP-01).
//!
//! Inverts [`crate::stft`]: each frame is transformed back to the time domain,
//! multiplied by the synthesis window, and overlap-added; the running sum of
//! squared windows compensates the analysis+synthesis windowing (the NOLA
//! condition — non-zero window-overlap-add — must hold for the division to be
//! stable). `center` trims the `n_fft/2` padding the forward transform added.
//!
//! `istft_streaming` (FR-OP-02: tail buffering, per-layer state carry-over) is
//! v0.5 and out of scope; the buffer layout here (contiguous overlap-add) does
//! not preclude a streaming variant that flushes completed samples.

use vokra_core::ir::graph::IstftAttrs;
use vokra_core::{Result, VokraError};

use crate::Spectrogram;
use crate::fft::{Complex32, FftPlan, RealFftPlan, norm_scale};
use crate::window::window;

/// Below this window-overlap energy a sample is treated as unreconstructable
/// (NOLA violation) and left at zero instead of dividing by ~0.
const NOLA_EPS: f32 = 1e-8;

/// Reconstructs a real signal from a complex [`Spectrogram`] under `attrs`.
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`] on a zero `n_fft` / `hop_length`, an
/// out-of-range `win_length`, or a `spectrogram.bins` that does not match the
/// expected `n_fft/2+1` (real input) or `n_fft` (complex input).
pub fn istft(spectrogram: &Spectrogram, attrs: &IstftAttrs) -> Result<Vec<f32>> {
    if attrs.n_fft == 0 || attrs.hop_length == 0 {
        return Err(VokraError::InvalidArgument(
            "istft: n_fft and hop_length must be non-zero".to_owned(),
        ));
    }
    if attrs.win_length == 0 || attrs.win_length > attrs.n_fft {
        return Err(VokraError::InvalidArgument(
            "istft: win_length must be in 1..=n_fft".to_owned(),
        ));
    }
    let n = attrs.n_fft;
    let expected_bins = if attrs.real_input { n / 2 + 1 } else { n };
    if spectrogram.bins != expected_bins {
        return Err(VokraError::InvalidArgument(format!(
            "istft: spectrogram has {} bins, expected {expected_bins}",
            spectrogram.bins
        )));
    }

    let hop = attrs.hop_length;
    let frames = spectrogram.frames;
    let synth_window = build_synth_window(attrs);
    let inv_scale = norm_scale(attrs.normalization, n, true);
    // Each stored bin = inv_scale · forward_raw; undo it before inverting.
    let unscale = if inv_scale == 0.0 {
        1.0
    } else {
        1.0 / inv_scale
    };

    let total = if frames > 0 {
        (frames - 1) * hop + n
    } else {
        0
    };
    let mut acc = vec![0.0f32; total];
    let mut wss = vec![0.0f32; total];

    let real_plan = attrs.real_input.then(|| RealFftPlan::new(n));
    let complex_plan = (!attrs.real_input).then(|| FftPlan::new(n));

    for f in 0..frames {
        let base = f * spectrogram.bins;
        let frame_time = if let Some(plan) = &real_plan {
            let half: Vec<Complex32> = (0..spectrogram.bins)
                .map(|b| {
                    Complex32::new(
                        spectrogram.re[base + b] * unscale,
                        spectrogram.im[base + b] * unscale,
                    )
                })
                .collect();
            plan.inverse(&half)
        } else {
            let full: Vec<Complex32> = (0..n)
                .map(|b| {
                    Complex32::new(
                        spectrogram.re[base + b] * unscale,
                        spectrogram.im[base + b] * unscale,
                    )
                })
                .collect();
            let raw = complex_plan
                .as_ref()
                .expect("complex plan")
                .inverse_raw(&full);
            let inv_n = 1.0 / n as f32;
            raw.iter().map(|c| c.re * inv_n).collect()
        };

        let start = f * hop;
        for i in 0..n {
            acc[start + i] += frame_time[i] * synth_window[i];
            wss[start + i] += synth_window[i] * synth_window[i];
        }
    }

    for (a, w) in acc.iter_mut().zip(&wss) {
        if *w > NOLA_EPS {
            *a /= *w;
        }
    }

    // Trim center padding, then honor an explicit target length.
    let (start, end) = if attrs.center {
        (n / 2, total.saturating_sub(n / 2))
    } else {
        (0, total)
    };
    let mut out = if start <= end {
        acc[start..end].to_vec()
    } else {
        Vec::new()
    };
    if let Some(len) = attrs.length {
        out.resize(len, 0.0);
    }
    Ok(out)
}

/// Builds the length-`n_fft` synthesis window (mirrors the analysis window).
fn build_synth_window(attrs: &IstftAttrs) -> Vec<f32> {
    let w = window(attrs.window, attrs.win_length, attrs.window_symmetry);
    if attrs.win_length == attrs.n_fft {
        return w;
    }
    let mut full = vec![0.0f32; attrs.n_fft];
    let offset = (attrs.n_fft - attrs.win_length) / 2;
    full[offset..offset + attrs.win_length].copy_from_slice(&w);
    full
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stft::stft;
    use vokra_core::ir::graph::StftAttrs;

    fn roundtrip_error(signal: &[f32], n_fft: usize, hop: usize, real_input: bool) -> f32 {
        let mut sa = StftAttrs::new(n_fft, hop);
        sa.real_input = real_input;
        let spec = stft(signal, &sa).unwrap();

        let mut ia = IstftAttrs::new(n_fft, hop);
        ia.real_input = real_input;
        ia.length = Some(signal.len());
        let recon = istft(&spec, &ia).unwrap();

        // Compare the interior (avoid the first/last frame edge effects).
        let guard = n_fft;
        let mut max = 0.0f32;
        for i in guard..signal.len() - guard {
            max = max.max((signal[i] - recon[i]).abs());
        }
        max
    }

    #[test]
    fn cola_roundtrip_real_input() {
        // Hann + hop = n_fft/4 satisfies COLA; reconstruction is near-exact.
        let signal: Vec<f32> = (0..4096)
            .map(|t| (t as f32 * 0.02).sin() + 0.3 * (t as f32 * 0.11).cos())
            .collect();
        let err = roundtrip_error(&signal, 512, 128, true);
        assert!(err < 1e-2, "reconstruction error {err}");
    }

    #[test]
    fn cola_roundtrip_complex_input() {
        let signal: Vec<f32> = (0..4096).map(|t| (t as f32 * 0.037).sin()).collect();
        let err = roundtrip_error(&signal, 256, 64, false);
        assert!(err < 1e-2, "reconstruction error {err}");
    }

    #[test]
    fn ortho_normalization_roundtrips() {
        use vokra_core::ir::graph::Normalization;
        let signal: Vec<f32> = (0..2048).map(|t| (t as f32 * 0.05).sin()).collect();
        let mut sa = StftAttrs::new(256, 64);
        sa.normalization = Normalization::Ortho;
        let spec = stft(&signal, &sa).unwrap();
        let mut ia = IstftAttrs::new(256, 64);
        ia.normalization = Normalization::Ortho;
        ia.length = Some(signal.len());
        let recon = istft(&spec, &ia).unwrap();
        let mut max = 0.0f32;
        for i in 256..signal.len() - 256 {
            max = max.max((signal[i] - recon[i]).abs());
        }
        assert!(max < 1e-2, "ortho reconstruction error {max}");
    }

    #[test]
    fn wrong_bin_count_is_rejected() {
        let spec = Spectrogram {
            frames: 2,
            bins: 99,
            re: vec![0.0; 2 * 99],
            im: vec![0.0; 2 * 99],
        };
        let attrs = IstftAttrs::new(256, 64); // expects 129 bins
        assert!(istft(&spec, &attrs).is_err());
    }

    #[test]
    fn length_override_sets_output_length() {
        let signal = vec![0.1f32; 3000];
        let sa = StftAttrs::new(256, 64);
        let spec = stft(&signal, &sa).unwrap();
        let mut ia = IstftAttrs::new(256, 64);
        ia.length = Some(1234);
        let recon = istft(&spec, &ia).unwrap();
        assert_eq!(recon.len(), 1234);
    }
}
