//! Mel filter bank (M0-04-T14; FR-OP-03, Slaney/HTK both supported).
//!
//! Reproduces `librosa.filters.mel`: triangular filters on FFT-bin
//! frequencies, band edges placed uniformly on the mel scale. Both the HTK warp
//! (`mel = 2595·log10(1 + f/700)`) and the Slaney auditory scale are provided
//! ([`MelScale`]), as is Slaney unit-area normalization ([`MelNorm`]). This is
//! the shape Whisper's front-end needs: `librosa.filters.mel(sr=16000,
//! n_fft=400, n_mels=80)` with the Slaney scale and Slaney norm.
//!
//! Frontend-spec parity (`vokra.frontend.*` bit-exact check, FR-LD-03) is M1;
//! this op only produces the matrix and applies it.

use vokra_core::ir::graph::{MelAttrs, MelInterp, MelNorm, MelScale};

/// A precomputed mel filter-bank matrix.
///
/// `weights` is row-major `[n_mels, n_freqs]` with `n_freqs = n_fft/2 + 1`.
#[derive(Debug, Clone)]
pub struct MelFilterbank {
    /// Number of mel bands (rows).
    pub n_mels: usize,
    /// Number of FFT frequency bins (columns), `n_fft/2 + 1`.
    pub n_freqs: usize,
    /// Filter weights, row-major `[n_mels, n_freqs]`.
    pub weights: Vec<f32>,
}

impl MelFilterbank {
    /// Builds the filter bank described by `attrs` (librosa-compatible).
    pub fn new(attrs: &MelAttrs) -> Self {
        let n_freqs = attrs.n_fft / 2 + 1;
        let n_mels = attrs.n_mels;
        let sr = attrs.sample_rate as f64;
        let fmax = attrs.fmax.map(|v| v as f64).unwrap_or(sr / 2.0);
        let fmin = attrs.fmin as f64;

        // FFT-bin center frequencies.
        let fft_freqs: Vec<f64> = (0..n_freqs)
            .map(|k| k as f64 * sr / attrs.n_fft as f64)
            .collect();

        // n_mels + 2 band edges, uniform on the mel scale. Both the mel and Hz
        // coordinates of every edge are kept: the ramp between edges is
        // interpolated in one domain or the other per `attrs.interp`.
        let min_mel = hz_to_mel(fmin, attrs.scale);
        let max_mel = hz_to_mel(fmax, attrs.scale);
        let band_mel: Vec<f64> = (0..n_mels + 2)
            .map(|i| min_mel + (max_mel - min_mel) * i as f64 / (n_mels + 1) as f64)
            .collect();
        let band_hz: Vec<f64> = band_mel
            .iter()
            .map(|&m| mel_to_hz(m, attrs.scale))
            .collect();
        // For the Kaldi (mel-domain) ramp, the FFT-bin mel coordinates are
        // reused across all bands; precompute them once.
        let fft_mel: Vec<f64> = fft_freqs
            .iter()
            .map(|&f| hz_to_mel(f, attrs.scale))
            .collect();

        let mut weights = vec![0.0f32; n_mels * n_freqs];
        for m in 0..n_mels {
            match attrs.interp {
                MelInterp::Hz => {
                    // librosa: triangle linear in Hz.
                    let left = band_hz[m];
                    let center = band_hz[m + 1];
                    let right = band_hz[m + 2];
                    let fdiff_low = center - left;
                    let fdiff_high = right - center;
                    for (k, &f) in fft_freqs.iter().enumerate() {
                        let lower = if fdiff_low > 0.0 {
                            (f - left) / fdiff_low
                        } else {
                            0.0
                        };
                        let upper = if fdiff_high > 0.0 {
                            (right - f) / fdiff_high
                        } else {
                            0.0
                        };
                        let w = lower.min(upper).max(0.0);
                        weights[m * n_freqs + k] = w as f32;
                    }
                }
                MelInterp::Mel => {
                    // Kaldi `MelBanks`: triangle linear in the mel domain, with
                    // strict `left_mel < mel(f) < right_mel` support (bins on
                    // the exact edges — including the Nyquist bin, which Kaldi's
                    // `num_fft_bins = n_fft/2` construction drops — get 0).
                    let left_mel = band_mel[m];
                    let center_mel = band_mel[m + 1];
                    let right_mel = band_mel[m + 2];
                    for (k, &mel) in fft_mel.iter().enumerate() {
                        let w = if mel > left_mel && mel < right_mel {
                            if mel <= center_mel {
                                (mel - left_mel) / (center_mel - left_mel)
                            } else {
                                (right_mel - mel) / (right_mel - center_mel)
                            }
                        } else {
                            0.0
                        };
                        weights[m * n_freqs + k] = w as f32;
                    }
                }
            }
            if matches!(attrs.norm, MelNorm::Slaney) {
                // Slaney: scale each filter to unit area (2 / bandwidth in Hz).
                // The bandwidth is the Hz distance between the outer edges,
                // independent of the ramp interpolation domain.
                let enorm = 2.0 / (band_hz[m + 2] - band_hz[m]);
                for k in 0..n_freqs {
                    weights[m * n_freqs + k] = (weights[m * n_freqs + k] as f64 * enorm) as f32;
                }
            }
        }

        Self {
            n_mels,
            n_freqs,
            weights,
        }
    }

    /// Projects a power (or magnitude) spectrogram onto the mel bands.
    ///
    /// `spectrum` is row-major `[frames, n_freqs]`; the result is row-major
    /// `[frames, n_mels]`.
    ///
    /// # Panics
    ///
    /// Panics if `spectrum.len() != frames * self.n_freqs`.
    pub fn apply(&self, spectrum: &[f32], frames: usize) -> Vec<f32> {
        assert_eq!(
            spectrum.len(),
            frames * self.n_freqs,
            "spectrogram shape mismatch"
        );
        let mut out = vec![0.0f32; frames * self.n_mels];
        for t in 0..frames {
            let frame = &spectrum[t * self.n_freqs..(t + 1) * self.n_freqs];
            for m in 0..self.n_mels {
                let filt = &self.weights[m * self.n_freqs..(m + 1) * self.n_freqs];
                let mut acc = 0.0f32;
                for (w, s) in filt.iter().zip(frame) {
                    acc += w * s;
                }
                out[t * self.n_mels + m] = acc;
            }
        }
        out
    }
}

/// Convenience constructor: [`MelFilterbank::new`].
pub fn mel_filterbank(attrs: &MelAttrs) -> MelFilterbank {
    MelFilterbank::new(attrs)
}

/// Hz → mel under the selected warp.
fn hz_to_mel(f: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + f / 700.0).log10(),
        MelScale::Slaney => {
            // librosa Slaney: linear below 1 kHz, log above.
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0;
            let min_log_mel = min_log_hz / f_sp; // = 15
            let logstep = (6.4f64).ln() / 27.0;
            if f >= min_log_hz {
                min_log_mel + (f / min_log_hz).ln() / logstep
            } else {
                f / f_sp
            }
        }
    }
}

/// Mel → Hz under the selected warp (inverse of [`hz_to_mel`]).
fn mel_to_hz(mel: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney => {
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0;
            let min_log_mel = min_log_hz / f_sp;
            let logstep = (6.4f64).ln() / 27.0;
            if mel >= min_log_mel {
                min_log_hz * (logstep * (mel - min_log_mel)).exp()
            } else {
                f_sp * mel
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn htk_mel_reference_points() {
        // HTK: mel(0)=0; mel(1000)=2595·log10(1+1000/700)=999.9856...
        assert!(hz_to_mel(0.0, MelScale::Htk).abs() < 1e-9);
        // 2595·log10(1 + 1000/700) = 999.98554 (standard HTK constant).
        let m1000 = hz_to_mel(1000.0, MelScale::Htk);
        assert!((m1000 - 999.985_54).abs() < 1e-2, "{m1000}");
        // Round-trip.
        for &f in &[100.0, 440.0, 2000.0, 8000.0] {
            let back = mel_to_hz(hz_to_mel(f, MelScale::Htk), MelScale::Htk);
            assert!((back - f).abs() < 1e-6);
        }
    }

    #[test]
    fn slaney_mel_is_linear_below_1khz() {
        // Below 1 kHz Slaney mel is f / (200/3).
        let f = 300.0;
        assert!((hz_to_mel(f, MelScale::Slaney) - f / (200.0 / 3.0)).abs() < 1e-9);
        // Round-trip above and below the knee.
        for &f in &[200.0, 999.0, 1000.0, 4000.0] {
            let back = mel_to_hz(hz_to_mel(f, MelScale::Slaney), MelScale::Slaney);
            assert!((back - f).abs() < 1e-6, "{f} -> {back}");
        }
    }

    #[test]
    fn filterbank_shape_and_nonnegativity() {
        let attrs = MelAttrs::new(16000, 400, 80);
        let fb = mel_filterbank(&attrs);
        assert_eq!(fb.n_mels, 80);
        assert_eq!(fb.n_freqs, 201);
        assert_eq!(fb.weights.len(), 80 * 201);
        assert!(fb.weights.iter().all(|&w| w >= 0.0));
        // Every mel filter has some support.
        for m in 0..80 {
            let row = &fb.weights[m * 201..(m + 1) * 201];
            assert!(row.iter().any(|&w| w > 0.0), "empty filter {m}");
        }
    }

    #[test]
    fn htk_filterbank_triangles_peak_near_one() {
        // Without Slaney normalization, triangular filters peak at 1.0.
        let attrs = MelAttrs {
            norm: MelNorm::None,
            scale: MelScale::Htk,
            ..MelAttrs::new(16000, 512, 40)
        };
        let fb = mel_filterbank(&attrs);
        for m in 0..fb.n_mels {
            let row = &fb.weights[m * fb.n_freqs..(m + 1) * fb.n_freqs];
            let peak = row.iter().cloned().fold(0.0f32, f32::max);
            assert!(peak <= 1.0 + 1e-6 && peak > 0.3, "filter {m} peak {peak}");
        }
    }

    #[test]
    fn apply_projects_frames() {
        let attrs = MelAttrs::new(16000, 400, 80);
        let fb = mel_filterbank(&attrs);
        let frames = 3;
        let spec = vec![1.0f32; frames * fb.n_freqs];
        let mel = fb.apply(&spec, frames);
        assert_eq!(mel.len(), frames * 80);
        assert!(mel.iter().all(|&v| v >= 0.0));
    }

    /// The exact CAM++ / CosyVoice Kaldi fbank filter geometry.
    fn kaldi_camplus_attrs() -> MelAttrs {
        MelAttrs {
            fmin: 20.0,
            fmax: Some(8000.0),
            scale: MelScale::Htk,
            norm: MelNorm::None,
            interp: MelInterp::Mel,
            ..MelAttrs::new(16000, 512, 80)
        }
    }

    #[test]
    fn kaldi_mel_triangles_peak_at_center_and_drop_nyquist() {
        // Mel-domain ramps: each filter peaks at 1.0 near its center mel edge,
        // and (matching Kaldi's `num_fft_bins = n_fft/2` construction) the
        // Nyquist bin `n_fft/2` carries zero weight in every band.
        let fb = mel_filterbank(&kaldi_camplus_attrs());
        assert_eq!(fb.n_mels, 80);
        assert_eq!(fb.n_freqs, 257);
        let nyq = 256;
        for m in 0..fb.n_mels {
            let row = &fb.weights[m * fb.n_freqs..(m + 1) * fb.n_freqs];
            let peak = row.iter().cloned().fold(0.0f32, f32::max);
            assert!(peak <= 1.0 + 1e-6 && peak > 0.3, "filter {m} peak {peak}");
            assert!(row.iter().all(|&w| (0.0..=1.0 + 1e-6).contains(&w)));
            assert_eq!(row[nyq], 0.0, "Nyquist bin must be dropped (filter {m})");
        }
        // Bin 0 (DC, 0 Hz < fmin=20) is excluded from every band.
        for m in 0..fb.n_mels {
            assert_eq!(fb.weights[m * fb.n_freqs], 0.0, "DC bin in filter {m}");
        }
    }

    #[test]
    fn kaldi_and_hz_interp_differ() {
        // Same edges, different ramp domain: the two modes must not coincide
        // (proves `MelInterp::Mel` is a genuinely distinct filter bank).
        let mel_mode = mel_filterbank(&kaldi_camplus_attrs());
        let hz_mode = mel_filterbank(&MelAttrs {
            interp: MelInterp::Hz,
            ..kaldi_camplus_attrs()
        });
        let max_diff = mel_mode
            .weights
            .iter()
            .zip(&hz_mode.weights)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff > 1e-3, "mel/Hz ramps unexpectedly identical");
    }
}
