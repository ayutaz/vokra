//! Mel-frequency cepstral coefficients (M0-04-T15; FR-OP-03).
//!
//! Composition op: `mfcc = dct-II( ln(max(mel, log_floor)) )`. It reuses the
//! [`crate::mel`] filter bank (so the Slaney/HTK choice flows through) and the
//! [`crate::dct`] DCT-II, keeping the first `n_mfcc` cepstral coefficients.
//!
//! Reference tools differ in the compression stage — librosa's `power_to_db`
//! uses `10·log10` with a dynamic-range clamp; matching a specific tool is a
//! parity-fixture setting (M0-04-T16). The natural-log definition here is fixed
//! and documented on [`MfccAttrs`].

use vokra_core::ir::graph::{DctAttrs, MfccAttrs};

use crate::dct::dct;
use crate::mel::MelFilterbank;

/// Computes MFCCs from a power (or magnitude) spectrogram.
///
/// `power` is row-major `[frames, n_freqs]` with `n_freqs = attrs.mel.n_fft/2 +
/// 1`; the result is row-major `[frames, attrs.n_mfcc]`.
///
/// # Panics
///
/// Panics if `power.len() != frames * (attrs.mel.n_fft/2 + 1)` or if
/// `attrs.n_mfcc` exceeds `attrs.mel.n_mels`.
pub fn mfcc(power: &[f32], frames: usize, attrs: &MfccAttrs) -> Vec<f32> {
    assert!(attrs.n_mfcc <= attrs.mel.n_mels, "n_mfcc exceeds n_mels");
    let fb = MelFilterbank::new(&attrs.mel);
    let mel = fb.apply(power, frames);

    // Natural-log compression with a floor to avoid ln(0).
    let floor = attrs.log_floor;
    let log_mel: Vec<f32> = mel.iter().map(|&v| v.max(floor).ln()).collect();

    let dct_attrs = DctAttrs {
        n_out: Some(attrs.n_mfcc),
        normalization: attrs.dct_norm,
    };
    dct(&log_mel, frames, attrs.mel.n_mels, &dct_attrs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::ir::graph::MelAttrs;

    #[test]
    fn mfcc_shape_is_frames_by_n_mfcc() {
        let mel = MelAttrs::new(16000, 400, 40);
        let attrs = MfccAttrs::new(mel, 13);
        let frames = 5;
        let n_freqs = 400 / 2 + 1;
        let power = vec![0.5f32; frames * n_freqs];
        let out = mfcc(&power, frames, &attrs);
        assert_eq!(out.len(), frames * 13);
    }

    #[test]
    fn mfcc_equals_manual_mel_log_dct() {
        // The composition must equal running the three stages by hand.
        let mel_attrs = MelAttrs::new(16000, 512, 32);
        let attrs = MfccAttrs::new(mel_attrs.clone(), 20);
        let frames = 4;
        let n_freqs = 512 / 2 + 1;
        let power: Vec<f32> = (0..frames * n_freqs)
            .map(|i| ((i % 7) as f32 + 1.0) * 0.1)
            .collect();

        let got = mfcc(&power, frames, &attrs);

        let fb = MelFilterbank::new(&mel_attrs);
        let mel = fb.apply(&power, frames);
        let log_mel: Vec<f32> = mel.iter().map(|&v| v.max(attrs.log_floor).ln()).collect();
        let expect = dct(
            &log_mel,
            frames,
            mel_attrs.n_mels,
            &DctAttrs {
                n_out: Some(20),
                normalization: attrs.dct_norm,
            },
        );
        assert_eq!(got.len(), expect.len());
        for (a, b) in got.iter().zip(&expect) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn constant_power_gives_flat_cepstrum() {
        // A flat mel spectrum (constant log) has near-zero higher cepstra.
        let mel_attrs = MelAttrs {
            norm: vokra_core::ir::graph::MelNorm::None,
            ..MelAttrs::new(16000, 512, 40)
        };
        let attrs = MfccAttrs::new(mel_attrs, 13);
        let frames = 2;
        let n_freqs = 512 / 2 + 1;
        // Constant spectrum -> each mel band integrates the same triangular
        // area; not perfectly flat, but c1.. should be far smaller than c0.
        let power = vec![1.0f32; frames * n_freqs];
        let out = mfcc(&power, frames, &attrs);
        let c0 = out[0].abs();
        let high: f32 = out[1..13].iter().map(|v| v.abs()).sum();
        assert!(c0 > high, "c0={c0} high-sum={high}");
    }
}
