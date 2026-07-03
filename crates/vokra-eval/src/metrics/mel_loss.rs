//! Log-mel L1 distance between two waveforms (reference-based TTS / vocoder
//! quality metric).
//!
//! The score is the mean absolute difference of the **log10 mel-power**
//! spectrograms of the hypothesis and the reference:
//!
//! ```text
//! logmel(x)[t, m] = log10(max(mel_power(x)[t, m], 1e-10))
//! mel_loss(h, r)  = mean_{t < min(Th, Tr), m} | logmel(h)[t, m] − logmel(r)[t, m] |
//! ```
//!
//! averaged over every `(frame, mel-band)` bin up to the shorter clip's frame
//! count (extra frames of the longer clip are ignored). Two identical inputs
//! score exactly `0.0`; adding energy that was not there before raises it.
//!
//! The STFT and mel bands come from explicit [`StftAttrs`] / [`MelAttrs`] — every
//! front-end knob is pinned (the CLAUDE.md `frontend_spec` pitfall: STFT ≠ FFT).
//! The spectrogram is computed with the *same* `vokra-ops` op path Whisper's
//! front-end uses (`stft` + `mel_filterbank`); the FFT and mel filter bank are
//! reused, never reimplemented here. A caller with a model's
//! `vokra_core::FrontendSpec` can therefore build a bit-exact front-end via
//! [`MelLoss::from_attrs`].

use super::{AudioRefMetric, Direction, Metric};
use vokra_core::ir::graph::{MelAttrs, StftAttrs};
use vokra_core::{Result, VokraError};
use vokra_ops::{mel_filterbank, stft};

/// Mel-spectrogram L1 loss (see the module docs for the exact definition).
#[derive(Debug, Clone)]
pub struct MelLoss {
    stft: StftAttrs,
    mel: MelAttrs,
}

impl MelLoss {
    /// Builds a mel-loss with librosa-style STFT/mel defaults for the given
    /// `sample_rate`, `n_fft`, `hop_length` and `n_mels` (Hann window, `center`
    /// reflect padding, Slaney mel scale + norm — see [`StftAttrs::new`] /
    /// [`MelAttrs::new`]).
    pub fn new(sample_rate: u32, n_fft: usize, hop_length: usize, n_mels: usize) -> Self {
        Self {
            stft: StftAttrs::new(n_fft, hop_length),
            mel: MelAttrs::new(sample_rate, n_fft, n_mels),
        }
    }

    /// Builds a mel-loss from explicit op attributes (e.g. derived from a
    /// model's `FrontendSpec` for a bit-exact front-end).
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if the STFT and mel `n_fft`
    /// disagree (the mel filter bank must match the spectrum it projects).
    pub fn from_attrs(stft: StftAttrs, mel: MelAttrs) -> Result<Self> {
        if stft.n_fft != mel.n_fft {
            return Err(VokraError::InvalidArgument(format!(
                "mel_loss: stft.n_fft ({}) != mel.n_fft ({})",
                stft.n_fft, mel.n_fft
            )));
        }
        Ok(Self { stft, mel })
    }

    /// The analysis sample rate (Hz) this metric is configured for.
    pub fn sample_rate(&self) -> u32 {
        self.mel.sample_rate
    }

    /// log10 mel-power spectrogram of `pcm`, row-major `[frames, n_mels]`,
    /// returned alongside the frame count.
    fn log_mel(&self, pcm: &[f32]) -> Result<(usize, Vec<f32>)> {
        let spec = stft(pcm, &self.stft)?;
        let power = spec.power();
        let fb = mel_filterbank(&self.mel);
        let mel = fb.apply(&power, spec.frames);
        let logmel: Vec<f32> = mel.iter().map(|&v| v.max(1e-10).log10()).collect();
        Ok((spec.frames, logmel))
    }

    /// Mean log-mel L1 loss between `hyp` and `reference` (both mono PCM at
    /// [`sample_rate`](Self::sample_rate)).
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if either clip is too short to
    /// yield a shared frame, and propagates any STFT error.
    pub fn loss(&self, hyp: &[f32], reference: &[f32]) -> Result<f64> {
        let (th, mh) = self.log_mel(hyp)?;
        let (tr, mr) = self.log_mel(reference)?;
        let frames = th.min(tr);
        let count = frames * self.mel.n_mels;
        if count == 0 {
            return Err(VokraError::InvalidArgument(
                "mel_loss: inputs too short to produce a shared frame".to_owned(),
            ));
        }
        // Both buffers are row-major `[frames_i, n_mels]`, so the first `count`
        // elements are exactly the shared `[min_frames, n_mels]` region.
        let sum: f64 = mh[..count]
            .iter()
            .zip(&mr[..count])
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .sum();
        Ok(sum / count as f64)
    }
}

impl Metric for MelLoss {
    fn name(&self) -> &str {
        "mel_loss"
    }
    fn direction(&self) -> Direction {
        Direction::LowerIsBetter
    }
}

impl AudioRefMetric for MelLoss {
    fn eval_audio(&self, hyp: &[f32], reference: &[f32], sample_rate: u32) -> Result<f64> {
        if sample_rate != self.mel.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "mel_loss: sample_rate {} != configured {} (frontend_spec must match)",
                sample_rate, self.mel.sample_rate
            )));
        }
        self.loss(hyp, reference)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 16 kHz, Whisper-like front-end — small enough to be fast in a unit test.
    const SR: u32 = 16_000;

    fn tone(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SR as f32).sin())
            .collect()
    }

    // Deterministic pseudo-noise in [-1, 1] — no RNG dependency (NFR-DS-02).
    fn noise(n: usize) -> Vec<f32> {
        let mut state: u32 = 0x1234_5678;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn add(a: &[f32], b: &[f32], scale: f32) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + scale * y).collect()
    }

    #[test]
    fn identical_inputs_score_exactly_zero() {
        let x = tone(440.0, 8_000);
        let ml = MelLoss::new(SR, 400, 160, 80);
        assert_eq!(ml.loss(&x, &x).unwrap(), 0.0);
    }

    #[test]
    fn perturbation_is_positive_and_grows_with_noise() {
        let x = tone(440.0, 8_000);
        let nz = noise(x.len());
        let ml = MelLoss::new(SR, 400, 160, 80);
        let small = ml.loss(&x, &add(&x, &nz, 0.01)).unwrap();
        let big = ml.loss(&x, &add(&x, &nz, 0.5)).unwrap();
        assert!(small > 0.0, "perturbed input must differ (got {small})");
        assert!(
            big > small,
            "more noise must score higher: {big} vs {small}"
        );
    }

    #[test]
    fn deterministic() {
        let x = tone(440.0, 8_000);
        let y = add(&x, &noise(x.len()), 0.1);
        let ml = MelLoss::new(SR, 400, 160, 80);
        assert_eq!(ml.loss(&x, &y).unwrap(), ml.loss(&x, &y).unwrap());
    }

    #[test]
    fn differing_lengths_use_the_shared_span_and_are_symmetric() {
        // Different-length clips must not panic: the loss is taken over the
        // shorter clip's frame span, and the absolute difference makes it
        // order-independent (loss(a,b) == loss(b,a)). (An exact 0 is NOT
        // expected even for an identical prefix — the STFT's center padding and
        // frame overlap make boundary frames differ between the two lengths.)
        let x = tone(440.0, 8_000);
        let mut longer = x.clone();
        longer.extend(tone(660.0, 4_000));
        let ml = MelLoss::new(SR, 400, 160, 80);
        let a = ml.loss(&x, &longer).unwrap();
        let b = ml.loss(&longer, &x).unwrap();
        assert!(a.is_finite() && a >= 0.0);
        assert_eq!(a, b, "mel_loss compares the shared span and is symmetric");
    }

    #[test]
    fn eval_audio_rejects_sample_rate_mismatch() {
        let x = tone(440.0, 8_000);
        let ml = MelLoss::new(SR, 400, 160, 80);
        assert!(ml.eval_audio(&x, &x, 22_050).is_err());
        assert_eq!(ml.eval_audio(&x, &x, SR).unwrap(), 0.0);
    }

    #[test]
    fn from_attrs_rejects_mismatched_n_fft() {
        let stft = StftAttrs::new(400, 160);
        let mel = MelAttrs::new(SR, 512, 80);
        assert!(MelLoss::from_attrs(stft, mel).is_err());
        // Matching n_fft builds fine and behaves like `new`.
        let ok = MelLoss::from_attrs(StftAttrs::new(400, 160), MelAttrs::new(SR, 400, 80)).unwrap();
        assert_eq!(ok.name(), "mel_loss");
        assert_eq!(ok.sample_rate(), SR);
    }

    #[test]
    fn zero_frame_input_errors_not_panics() {
        // With center padding even empty audio yields frames, so the count==0
        // guard is reached only when framing genuinely produces no frame:
        // center=false with a signal shorter than one FFT window. That must be
        // a clean error, never a panic or a divide-by-zero.
        let mut stft = StftAttrs::new(400, 160);
        stft.center = false;
        let ml = MelLoss::from_attrs(stft, MelAttrs::new(SR, 400, 80)).unwrap();
        let short = tone(440.0, 100); // < n_fft = 400 → zero frames
        assert!(ml.loss(&short, &short).is_err());
    }
}
