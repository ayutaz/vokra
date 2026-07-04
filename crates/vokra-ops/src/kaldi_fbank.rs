//! Kaldi-compatible log-mel fbank + CMN front-end (M0-08, CAM++ speaker encoder).
//!
//! The CosyVoice / 3D-Speaker CAM++ speaker encoder is fed the *Kaldi* fbank —
//! not the librosa/Whisper log-mel this crate already produces. The two differ
//! in several load-bearing details, all reproduced here:
//!
//! - **snip-edges framing** (no center padding): `n_frames = 1 + (N −
//!   frame_length) / frame_shift`, frame `f` starting at `f · frame_shift`;
//! - **per-frame** DC-offset removal then pre-emphasis (`0.97`), applied to
//!   each frame independently (librosa applies neither, or applies them once to
//!   the whole utterance);
//! - the **Povey window** ([`povey`], Hann^0.85);
//! - a power spectrum over a **power-of-two padded** FFT (`400 → 512`);
//! - a **Kaldi HTK mel** with **mel-domain** triangular ramps
//!   ([`MelInterp::Mel`]) and *no* Slaney area normalization;
//! - `log(max(e, ε))`;
//! - **CMN**: subtract each bin's per-utterance time-mean (CosyVoice default).
//!
//! # Validation status (M0-08 stage 3)
//!
//! No Kaldi/torchaudio fbank oracle is installed, so this front-end is validated
//! **structurally only** (shapes, determinism, sane/finite ranges, the
//! per-frame-DC + CMN invariants) — see the tests below. **Bit-exact agreement
//! with `torchaudio.compliance.kaldi.fbank` / Kaldi `compute-fbank-feats` is a
//! deferred follow-up** that needs an offline oracle to be installed (design
//! stage B). The individual pieces are faithful to the Kaldi source
//! (`feature-window.cc`, `feature-fbank.cc`, `mel-computations.cc`); no
//! reference numbers are fabricated.

use vokra_core::ir::graph::{MelAttrs, MelInterp, MelNorm, MelScale};
use vokra_core::{Result, VokraError};

use crate::fft::RealFftPlan;
use crate::mel::MelFilterbank;
use crate::window::povey;

/// Kaldi fbank knobs (mirrors `torchaudio.compliance.kaldi.fbank`), with the
/// frame geometry expressed directly in **samples** to avoid the ms↔samples
/// rounding ambiguity.
#[derive(Debug, Clone, PartialEq)]
pub struct KaldiFbankOpts {
    /// Audio sample rate, Hz.
    pub sample_rate: u32,
    /// Number of mel bands (output feature dimension).
    pub num_mel_bins: usize,
    /// Analysis frame length, in samples (Kaldi `frame_length` × sr / 1000).
    pub frame_length: usize,
    /// Hop between frames, in samples (Kaldi `frame_shift` × sr / 1000).
    pub frame_shift: usize,
    /// Subtract the per-frame mean before pre-emphasis (Kaldi
    /// `remove_dc_offset`).
    pub remove_dc_offset: bool,
    /// First-order pre-emphasis coefficient applied per frame (`0.0` = off).
    pub preemph_coeff: f32,
    /// Low mel band edge, Hz (Kaldi `low_freq`).
    pub low_freq: f32,
    /// High mel band edge, Hz; `<= 0.0` means `nyquist + high_freq` (Kaldi
    /// `high_freq` convention — `0.0` ⇒ the Nyquist frequency).
    pub high_freq: f32,
    /// Use the power spectrum `|X|²` (`true`) or the magnitude `|X|` (`false`).
    pub use_power: bool,
    /// Take `log(max(e, ε))` of the mel energies (Kaldi `use_log_fbank`).
    pub use_log: bool,
    /// Subtract each bin's per-utterance time-mean (CMN; Kaldi `subtract_mean`).
    pub subtract_mean: bool,
    /// Zero-pad each frame up to the next power of two before the FFT (Kaldi
    /// `round_to_power_of_two`); `400 → 512`.
    pub round_to_power_of_two: bool,
}

impl KaldiFbankOpts {
    /// The exact CosyVoice / 3D-Speaker CAM++ fbank configuration: 16 kHz, 80
    /// bins, 25 ms / 10 ms frames (`400` / `160` samples), per-frame DC removal
    /// then `0.97` pre-emphasis, dither off, power spectrum, Kaldi HTK mel over
    /// the `20`–`8000` Hz band, log, and per-utterance CMN.
    pub fn camplus() -> Self {
        Self {
            sample_rate: 16_000,
            num_mel_bins: 80,
            frame_length: 400,
            frame_shift: 160,
            remove_dc_offset: true,
            preemph_coeff: 0.97,
            low_freq: 20.0,
            high_freq: 0.0,
            use_power: true,
            use_log: true,
            subtract_mean: true,
            round_to_power_of_two: true,
        }
    }
}

/// Computes Kaldi fbank features from mono PCM at `opts.sample_rate`.
///
/// Returns `(feats, n_frames)` where `feats` is row-major `[n_frames,
/// num_mel_bins]` (frame-major) — the layout [`SpeakerEncoder::embed`] expects.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] if `frame_length` / `frame_shift` are zero,
/// or if `pcm` is shorter than one frame (snip-edges yields no frames — the
/// reference clip is too short to embed).
pub fn kaldi_fbank(pcm: &[f32], opts: &KaldiFbankOpts) -> Result<(Vec<f32>, usize)> {
    let (flen, fshift) = (opts.frame_length, opts.frame_shift);
    if flen == 0 || fshift == 0 {
        return Err(VokraError::InvalidArgument(
            "kaldi_fbank: frame_length and frame_shift must be > 0".into(),
        ));
    }
    // Snip-edges framing (no center pad): a frame must fit entirely in the
    // signal, so a clip shorter than one frame has no features at all.
    if pcm.len() < flen {
        return Err(VokraError::InvalidArgument(format!(
            "kaldi_fbank: reference audio too short — {} samples < frame_length {flen} \
             (need ≥ {flen} samples ≈ {:.0} ms at {} Hz)",
            pcm.len(),
            1000.0 * flen as f64 / opts.sample_rate.max(1) as f64,
            opts.sample_rate
        )));
    }
    let n_frames = 1 + (pcm.len() - flen) / fshift;

    let fft_size = if opts.round_to_power_of_two {
        flen.next_power_of_two()
    } else {
        flen
    };
    let n_freqs = fft_size / 2 + 1;
    let nbins = opts.num_mel_bins;

    // Analysis window, FFT plan and mel bank are frame-invariant: build once.
    let win = povey(flen);
    let plan = RealFftPlan::new(fft_size);
    let nyquist = 0.5 * opts.sample_rate as f32;
    let fmax = if opts.high_freq > 0.0 {
        opts.high_freq
    } else {
        nyquist + opts.high_freq
    };
    let fb = MelFilterbank::new(&MelAttrs {
        sample_rate: opts.sample_rate,
        n_fft: fft_size,
        n_mels: nbins,
        fmin: opts.low_freq,
        fmax: Some(fmax),
        scale: MelScale::Htk,
        norm: MelNorm::None,
        interp: MelInterp::Mel,
    });
    debug_assert_eq!(fb.n_freqs, n_freqs);

    let mut out = vec![0.0f32; n_frames * nbins];
    // Frame scratch is `fft_size` long; the `[flen, fft_size)` tail is the
    // zero pad and is never written, so it stays zero across frames.
    let mut frame = vec![0.0f32; fft_size];
    let mut power = vec![0.0f32; n_freqs];

    for f in 0..n_frames {
        let start = f * fshift;
        frame[..flen].copy_from_slice(&pcm[start..start + flen]);

        // (1) Per-frame DC removal (mean accumulated in f64 for stability).
        if opts.remove_dc_offset {
            let mean =
                (frame[..flen].iter().map(|&x| f64::from(x)).sum::<f64>() / flen as f64) as f32;
            for v in &mut frame[..flen] {
                *v -= mean;
            }
        }
        // (2) Pre-emphasis, Kaldi in-place backward recurrence:
        //     y[i] -= c·y[i-1] (i high→low, so y[i-1] is still the input),
        //     y[0] -= c·y[0]  ⇒  y[0] = (1−c)·y[0].
        let c = opts.preemph_coeff;
        if c != 0.0 {
            for i in (1..flen).rev() {
                frame[i] -= c * frame[i - 1];
            }
            frame[0] -= c * frame[0];
        }
        // (3) Windowing (Povey).
        for (v, &w) in frame[..flen].iter_mut().zip(win.iter()) {
            *v *= w;
        }
        // (4) Power (or magnitude) spectrum of the zero-padded frame.
        let spec = plan.forward(&frame);
        for (p, cx) in power.iter_mut().zip(&spec) {
            let e = cx.re * cx.re + cx.im * cx.im;
            *p = if opts.use_power { e } else { e.sqrt() };
        }
        // (5) Mel projection + log floor.
        let mel = fb.apply(&power, 1);
        for (o, &e) in out[f * nbins..(f + 1) * nbins].iter_mut().zip(&mel) {
            *o = if opts.use_log {
                e.max(f32::EPSILON).ln()
            } else {
                e
            };
        }
    }

    // (6) CMN: subtract each bin's mean over time (per-utterance).
    if opts.subtract_mean {
        for b in 0..nbins {
            let mut s = 0.0f64;
            for f in 0..n_frames {
                s += f64::from(out[f * nbins + b]);
            }
            let m = (s / n_frames as f64) as f32;
            for f in 0..n_frames {
                out[f * nbins + b] -= m;
            }
        }
    }

    Ok((out, n_frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random-ish speech-like signal of `n` samples.
    fn signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|k| {
                let t = k as f32;
                0.6 * (t * 0.05).sin() + 0.3 * (t * 0.017).sin() + 0.1 * (t * 0.3).cos()
            })
            .collect()
    }

    #[test]
    fn frame_count_and_shape_follow_snip_edges() {
        let opts = KaldiFbankOpts::camplus();
        // 1 s at 16 kHz: n_frames = 1 + (16000-400)/160 = 98.
        let (feats, t) = kaldi_fbank(&signal(16_000), &opts).unwrap();
        assert_eq!(t, 1 + (16_000 - 400) / 160);
        assert_eq!(t, 98);
        assert_eq!(feats.len(), t * 80);
        // Exactly one frame when the clip is exactly one frame long.
        let (_f1, t1) = kaldi_fbank(&signal(400), &opts).unwrap();
        assert_eq!(t1, 1);
        // 401 samples still yields a single frame (401-400 < shift).
        let (_f2, t2) = kaldi_fbank(&signal(401), &opts).unwrap();
        assert_eq!(t2, 1);
    }

    #[test]
    fn deterministic_and_finite() {
        let opts = KaldiFbankOpts::camplus();
        let x = signal(8_000);
        let (a, ta) = kaldi_fbank(&x, &opts).unwrap();
        let (b, tb) = kaldi_fbank(&x, &opts).unwrap();
        assert_eq!(ta, tb);
        assert_eq!(a, b, "front-end must be bit-for-bit deterministic");
        assert!(a.iter().all(|v| v.is_finite()), "non-finite fbank value");
    }

    #[test]
    fn too_short_input_is_an_error() {
        let opts = KaldiFbankOpts::camplus();
        assert!(kaldi_fbank(&signal(399), &opts).is_err());
        assert!(kaldi_fbank(&[], &opts).is_err());
    }

    #[test]
    fn cmn_zeroes_each_bin_time_mean() {
        // After per-utterance CMN, every bin's mean over frames is ~0.
        let opts = KaldiFbankOpts::camplus();
        let (feats, t) = kaldi_fbank(&signal(20_000), &opts).unwrap();
        for b in 0..80 {
            let mean: f64 = (0..t).map(|f| f64::from(feats[f * 80 + b])).sum::<f64>() / t as f64;
            assert!(mean.abs() < 1e-3, "bin {b} residual mean {mean}");
        }
    }

    #[test]
    fn constant_signal_collapses_to_zero_after_dc_and_cmn() {
        // A constant (pure-DC) signal: per-frame DC removal zeros every frame,
        // so every bin is the same log-floor constant, and CMN drives the whole
        // feature matrix to ~0 — exercising the DC-removal + CMN invariants
        // together without needing a numeric oracle.
        let opts = KaldiFbankOpts::camplus();
        let (feats, _t) = kaldi_fbank(&vec![0.42f32; 6_000], &opts).unwrap();
        for &v in &feats {
            assert!(v.abs() < 1e-3, "constant input left residue {v}");
        }
    }

    #[test]
    fn without_cmn_a_tone_has_finite_structured_output() {
        // No CMN: values are raw log-mel energies. A 1 kHz tone must produce a
        // non-flat spectrum (some bins clearly larger than others) and stay
        // finite — a sanity check that the mel projection is wired correctly.
        let opts = KaldiFbankOpts {
            subtract_mean: false,
            ..KaldiFbankOpts::camplus()
        };
        let sr = 16_000.0f32;
        let tone: Vec<f32> = (0..8_000)
            .map(|k| (2.0 * std::f32::consts::PI * 1000.0 * k as f32 / sr).sin())
            .collect();
        let (feats, t) = kaldi_fbank(&tone, &opts).unwrap();
        assert!(feats.iter().all(|v| v.is_finite()));
        // First frame: the spread across bins must be non-trivial.
        let row = &feats[..80];
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in row {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(
            hi - lo > 1.0,
            "tone spectrum unexpectedly flat ({lo}..{hi})"
        );
        assert!(t > 0);
    }
}
