//! Whisper log-mel front-end (PCM → `[n_mels, 3000]` log-mel features).
//!
//! Reuses the M0-04 `vokra-ops` STFT and mel filter bank (FR-OP-01/03) and
//! reproduces the openai/whisper `log_mel_spectrogram` post-processing exactly
//! (HF `WhisperFeatureExtractor` matches it):
//!
//! 1. zero-pad (or trim) the mono PCM to exactly 30 s = [`N_SAMPLES`] samples;
//! 2. STFT with `n_fft = 400`, `hop = 160`, periodic Hann, `center = true`
//!    reflect padding, no FFT normalization (raw); take the power `|X|²` and
//!    **drop the last STFT frame**, leaving [`N_FRAMES`] = 3000 frames;
//! 3. project onto `n_mels` Slaney-scale / Slaney-norm mel bands;
//! 4. `log10(clamp(·, 1e-10))`, then dynamic-range compress to the global
//!    max minus 8, then `(· + 4) / 4`.
//!
//! Parameters here come from Whisper's fixed front-end (`n_fft`, `hop`,
//! `sample_rate`, Slaney mel) — the same values the converter writes into
//! `vokra.frontend.*`. The bit-exact `frontend_spec` **check** (FR-LD-03) is
//! M1-03; this WP only reproduces the front-end (STFT ≠ FFT — every knob is
//! explicit, per the CLAUDE.md pitfall).

use vokra_core::ir::graph::{MelAttrs, StftAttrs};
use vokra_ops::{mel_filterbank, stft};

/// Model sample rate in Hz (Whisper is 16 kHz).
pub const SAMPLE_RATE: u32 = 16_000;
/// STFT window / FFT size.
pub const N_FFT: usize = 400;
/// STFT hop length.
pub const HOP: usize = 160;
/// Fixed input length: 30 s at 16 kHz.
pub const N_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
/// Number of log-mel frames after dropping the trailing STFT frame.
pub const N_FRAMES: usize = 3000;

/// Computes the `[n_mels, N_FRAMES]` (row-major) log-mel features of mono
/// `pcm`, assumed to already be at [`SAMPLE_RATE`].
///
/// The input is zero-padded or trimmed to 30 s, so the output frame count is
/// always [`N_FRAMES`] regardless of the input length.
pub fn log_mel(pcm: &[f32], n_mels: usize) -> Vec<f32> {
    // 1. Pad / trim to exactly 30 s.
    let mut buf = vec![0.0f32; N_SAMPLES];
    let n = pcm.len().min(N_SAMPLES);
    buf[..n].copy_from_slice(&pcm[..n]);

    // 2. STFT → power, drop the trailing frame.
    let spec = stft(&buf, &StftAttrs::new(N_FFT, HOP)).expect("valid whisper STFT attrs");
    let bins = spec.bins; // n_fft/2 + 1 = 201
    let frames = spec.frames.min(N_FRAMES + 1);
    let kept = frames.min(N_FRAMES);
    let power = spec.power();

    // 3. Mel projection on the kept frames → [kept, n_mels].
    let fb = mel_filterbank(&MelAttrs::new(SAMPLE_RATE, N_FFT, n_mels));
    let mel = fb.apply(&power[..kept * bins], kept);

    // 4. log10 + dynamic-range compression + normalization, transposed to
    //    [n_mels, N_FRAMES]. Frames beyond `kept` (only possible for absurdly
    //    short inputs) stay at the log floor.
    let floor_log = 1e-10f32.log10();
    let mut out = vec![floor_log; n_mels * N_FRAMES];
    let mut gmax = f32::NEG_INFINITY;
    for t in 0..kept {
        for m in 0..n_mels {
            let l = mel[t * n_mels + m].max(1e-10).log10();
            out[m * N_FRAMES + t] = l;
            if l > gmax {
                gmax = l;
            }
        }
    }
    let dyn_floor = gmax - 8.0;
    for v in &mut out {
        *v = (v.max(dyn_floor) + 4.0) / 4.0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_shape_is_n_mels_by_n_frames() {
        let pcm = vec![0.0f32; SAMPLE_RATE as usize]; // 1 s of silence
        let out = log_mel(&pcm, 80);
        assert_eq!(out.len(), 80 * N_FRAMES);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_tone_produces_finite_bounded_features() {
        // 440 Hz tone; log-mel is normalized to a bounded range by construction.
        let pcm: Vec<f32> = (0..SAMPLE_RATE as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let out = log_mel(&pcm, 80);
        // After (x+4)/4 with x in [max-8, max] and Whisper's normalization the
        // features sit within roughly [-1, 1]; assert a generous finite bound.
        assert!(out.iter().all(|v| v.is_finite() && *v > -2.0 && *v < 2.0));
    }

    #[test]
    fn empty_input_pads_to_full_frame_grid() {
        // The pad branch with a zero-length slice: buf stays all-zero, so the
        // output is still the full [n_mels, N_FRAMES] grid with finite values.
        let out = log_mel(&[], 80);
        assert_eq!(out.len(), 80 * N_FRAMES);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn oversized_input_is_trimmed_to_full_frame_grid() {
        // Longer than 30 s exercises the trim branch (pcm.len() > N_SAMPLES); the
        // frame count is still fixed and every value stays finite.
        let pcm = vec![0.1f32; N_SAMPLES + 5000];
        let out = log_mel(&pcm, 80);
        assert_eq!(out.len(), 80 * N_FRAMES);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
