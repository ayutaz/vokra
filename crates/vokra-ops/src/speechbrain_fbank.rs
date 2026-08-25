//! SpeechBrain's native filter-bank frontend.
//!
//! SpeechBrain's [`Fbank`](https://github.com/speechbrain/speechbrain/blob/e5cb1f65b940634215650aa1171e0440d0808123/speechbrain/lobes/features.py)
//! is not Kaldi fbank and is not librosa mel.  It combines a centered STFT,
//! power spectrum, HTK-spaced triangular filters whose two slopes use the
//! lower adjacent band width, decibel conversion with a per-utterance dynamic
//! range floor, and optional sentence mean normalization.  Keeping this as a
//! named operator prevents ECAPA/X-vector/language-ID models from silently
//! selecting a merely similar frontend.

use vokra_core::ir::graph::{Normalization, PadMode, StftAttrs, Window, WindowSymmetry};
use vokra_core::{Result, VokraError};

use crate::stft;

/// Explicit attributes for the SpeechBrain filter-bank frontend.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechbrainFbankAttrs {
    /// PCM sample rate in Hz.
    pub sample_rate: u32,
    /// Complete STFT contract, including window and padding convention.
    pub stft: StftAttrs,
    /// Number of output triangular filters.
    pub n_mels: usize,
    /// Lowest filter edge in Hz.
    pub f_min: f32,
    /// Highest filter edge in Hz.
    pub f_max: f32,
    /// Linear-energy floor before `10 * log10`.
    pub amin: f32,
    /// Per-utterance dynamic range in decibels.
    pub top_db: f32,
    /// Subtract each mel channel's utterance mean after dB conversion.
    pub sentence_mean_norm: bool,
}

impl SpeechbrainFbankAttrs {
    fn voxceleb(n_mels: usize) -> Self {
        let mut stft = StftAttrs::new(400, 160);
        stft.win_length = 400;
        stft.window = Window::Hamming;
        stft.window_symmetry = WindowSymmetry::Periodic;
        stft.center = true;
        stft.pad_mode = PadMode::Constant;
        stft.normalization = Normalization::Backward;
        stft.causal = false;
        stft.real_input = true;
        Self {
            sample_rate: 16_000,
            stft,
            n_mels,
            f_min: 0.0,
            f_max: 8_000.0,
            amin: 1.0e-10,
            top_db: 80.0,
            sentence_mean_norm: true,
        }
    }

    /// Exact frontend used by `speechbrain/spkrec-xvect-voxceleb`.
    #[must_use]
    pub fn xvector_voxceleb() -> Self {
        Self::voxceleb(24)
    }

    /// Exact 80-bin frontend used by
    /// `speechbrain/spkrec-ecapa-voxceleb`.
    ///
    /// SpeechBrain Lang-ID variants reuse the same frontend algorithm but
    /// carry their own checked mel width (60 for VoxLingua107, 80 for
    /// CommonLanguage); their runtime binder updates only `n_mels` after
    /// validating the prepared-v2 artifact contract.
    #[must_use]
    pub fn ecapa_voxceleb() -> Self {
        Self::voxceleb(80)
    }
}

/// Computes row-major `[frames, n_mels]` SpeechBrain filter-bank features.
///
/// The returned frame count is always consistent with the flattened output.
/// No resampling is performed; callers must verify the PCM rate against
/// [`SpeechbrainFbankAttrs::sample_rate`].
pub fn speechbrain_fbank(pcm: &[f32], attrs: &SpeechbrainFbankAttrs) -> Result<(Vec<f32>, usize)> {
    validate(attrs)?;
    if pcm.is_empty() {
        return Err(VokraError::InvalidArgument(
            "speechbrain_fbank: empty PCM input".to_owned(),
        ));
    }

    let spectrum = stft(pcm, &attrs.stft)?;
    let power = spectrum.power();
    let filters = filter_matrix(attrs, spectrum.bins);
    let mut features = vec![0.0f32; spectrum.frames * attrs.n_mels];

    for frame in 0..spectrum.frames {
        let bins = &power[frame * spectrum.bins..(frame + 1) * spectrum.bins];
        for mel in 0..attrs.n_mels {
            let weights = &filters[mel * spectrum.bins..(mel + 1) * spectrum.bins];
            let energy = bins
                .iter()
                .zip(weights)
                .fold(0.0f32, |sum, (&value, &weight)| sum + value * weight);
            features[frame * attrs.n_mels + mel] = 10.0 * energy.max(attrs.amin).log10();
        }
    }

    let floor = features.iter().copied().fold(f32::NEG_INFINITY, f32::max) - attrs.top_db;
    for value in &mut features {
        *value = value.max(floor);
    }

    if attrs.sentence_mean_norm {
        for mel in 0..attrs.n_mels {
            let mean = (0..spectrum.frames)
                .map(|frame| features[frame * attrs.n_mels + mel])
                .sum::<f32>()
                / spectrum.frames as f32;
            for frame in 0..spectrum.frames {
                features[frame * attrs.n_mels + mel] -= mean;
            }
        }
    }

    Ok((features, spectrum.frames))
}

fn validate(attrs: &SpeechbrainFbankAttrs) -> Result<()> {
    if attrs.sample_rate == 0 || attrs.n_mels == 0 {
        return Err(VokraError::InvalidArgument(
            "speechbrain_fbank: sample_rate and n_mels must be non-zero".to_owned(),
        ));
    }
    let nyquist = attrs.sample_rate as f32 * 0.5;
    if !attrs.f_min.is_finite()
        || !attrs.f_max.is_finite()
        || attrs.f_min < 0.0
        || attrs.f_min >= attrs.f_max
        || attrs.f_max > nyquist
    {
        return Err(VokraError::InvalidArgument(format!(
            "speechbrain_fbank: require 0 <= f_min < f_max <= Nyquist ({nyquist}), got {}..{}",
            attrs.f_min, attrs.f_max
        )));
    }
    if !attrs.amin.is_finite() || attrs.amin <= 0.0 {
        return Err(VokraError::InvalidArgument(
            "speechbrain_fbank: amin must be finite and positive".to_owned(),
        ));
    }
    if !attrs.top_db.is_finite() || attrs.top_db < 0.0 {
        return Err(VokraError::InvalidArgument(
            "speechbrain_fbank: top_db must be finite and non-negative".to_owned(),
        ));
    }
    if !attrs.stft.real_input {
        return Err(VokraError::InvalidArgument(
            "speechbrain_fbank: STFT must use a real-input half spectrum".to_owned(),
        ));
    }
    Ok(())
}

fn filter_matrix(attrs: &SpeechbrainFbankAttrs, bins: usize) -> Vec<f32> {
    debug_assert_eq!(bins, attrs.stft.n_fft / 2 + 1);
    let min_mel = hz_to_mel(attrs.f_min);
    let max_mel = hz_to_mel(attrs.f_max);
    let edges = (0..attrs.n_mels + 2)
        .map(|index| {
            let mel = min_mel + (max_mel - min_mel) * index as f32 / (attrs.n_mels + 1) as f32;
            mel_to_hz(mel)
        })
        .collect::<Vec<_>>();
    let mut output = vec![0.0f32; attrs.n_mels * bins];
    for mel in 0..attrs.n_mels {
        let center = edges[mel + 1];
        // This deliberate lower-band reuse is SpeechBrain's `band[:-1]`
        // contract, not the conventional independent left/right edge widths.
        let band = edges[mel + 1] - edges[mel];
        for bin in 0..bins {
            let frequency = bin as f32 * attrs.sample_rate as f32 / attrs.stft.n_fft as f32;
            let slope = (frequency - center) / band;
            output[mel * bins + bin] = (slope + 1.0).min(1.0 - slope).max(0.0);
        }
    }
    output
}

fn hz_to_mel(hz: f32) -> f32 {
    2_595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2_595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xvector_frontend_has_the_official_axes() {
        let attrs = SpeechbrainFbankAttrs::xvector_voxceleb();
        assert_eq!(attrs.sample_rate, 16_000);
        assert_eq!(attrs.stft.n_fft, 400);
        assert_eq!(attrs.stft.hop_length, 160);
        assert_eq!(attrs.stft.window, Window::Hamming);
        assert_eq!(attrs.stft.pad_mode, PadMode::Constant);
        assert_eq!(attrs.n_mels, 24);
    }

    #[test]
    fn ecapa_frontend_differs_only_in_mel_width() {
        let xvector = SpeechbrainFbankAttrs::xvector_voxceleb();
        let ecapa = SpeechbrainFbankAttrs::ecapa_voxceleb();
        assert_eq!(ecapa.sample_rate, 16_000);
        assert_eq!(ecapa.stft, xvector.stft);
        assert_eq!(ecapa.n_mels, 80);
        assert!(ecapa.sentence_mean_norm);
    }

    #[test]
    fn one_second_signal_produces_101_finite_frames() {
        let pcm = (0..16_000)
            .map(|index| {
                let phase = index as f32 * 2.0 * std::f32::consts::PI * 440.0 / 16_000.0;
                phase.sin()
            })
            .collect::<Vec<_>>();
        let (features, frames) =
            speechbrain_fbank(&pcm, &SpeechbrainFbankAttrs::xvector_voxceleb()).unwrap();
        assert_eq!(frames, 101);
        assert_eq!(features.len(), frames * 24);
        assert!(features.iter().all(|value| value.is_finite()));
        for mel in 0..24 {
            let mean = (0..frames)
                .map(|frame| f64::from(features[frame * 24 + mel]))
                .sum::<f64>()
                / frames as f64;
            // Accumulate the invariant check in f64 so the assertion does not
            // add a second f32 reduction error. The production normalization
            // deliberately matches SpeechBrain's f32 mean; its measured
            // residual on 101 frames is below 5e-5.
            assert!(mean.abs() < 5.0e-5, "mel {mel} mean {mean}");
        }
    }

    #[test]
    fn empty_input_is_rejected() {
        let error = speechbrain_fbank(&[], &SpeechbrainFbankAttrs::xvector_voxceleb()).unwrap_err();
        assert!(error.to_string().contains("empty PCM"));
    }
}
