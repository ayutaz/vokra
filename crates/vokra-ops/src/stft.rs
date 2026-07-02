//! Short-time Fourier transform (M0-04-T10, T11; FR-OP-01).
//!
//! STFT ≠ FFT (CLAUDE.md pitfall): the transform is *window + framing +
//! normalization + causal mode* around an FFT. Each of those is an explicit
//! [`StftAttrs`] knob rather than an implicit default:
//!
//! - `center` / `pad_mode`: symmetric padding by `n_fft/2` so frame `t` is
//!   centered on sample `t·hop` (matches `torch.stft(center=True)`);
//! - `causal`: no look-ahead — left-pad only, so truncating future samples
//!   never changes an already-emitted frame;
//! - `win_length < n_fft`: the window is centered and zero-padded to `n_fft`;
//! - `normalization`: forward/backward/ortho scaling (`Ortho` == torch
//!   `normalized=True`, i.e. `1/√n_fft`);
//! - `real_input`: RFFT half-spectrum (`n_fft/2+1` bins) vs the full complex
//!   spectrum.
//!
//! Output layout is row-major `[frames, bins]`, ready for the mel front-end
//! (M0-06 Whisper). Streaming iSTFT tail state (FR-OP-02) is v0.5.

use vokra_core::ir::graph::{PadMode, StftAttrs};
use vokra_core::{Result, VokraError};

use crate::fft::{Complex32, FftPlan, RealFftPlan, norm_scale};
use crate::window::window;

/// A complex spectrogram, row-major `[frames, bins]`.
///
/// `re` and `im` are the split real/imaginary parts (the complex value type
/// stays internal — public `complex64` on the IR is FR-EX-09, out of M0 scope).
#[derive(Debug, Clone)]
pub struct Spectrogram {
    /// Number of time frames (rows).
    pub frames: usize,
    /// Number of frequency bins per frame (columns).
    pub bins: usize,
    /// Real parts, row-major `[frames, bins]`.
    pub re: Vec<f32>,
    /// Imaginary parts, row-major `[frames, bins]`.
    pub im: Vec<f32>,
}

impl Spectrogram {
    /// The power spectrogram `|X|² = re² + im²`, row-major `[frames, bins]`.
    pub fn power(&self) -> Vec<f32> {
        self.re
            .iter()
            .zip(&self.im)
            .map(|(r, i)| r * r + i * i)
            .collect()
    }

    /// The magnitude spectrogram `|X|`, row-major `[frames, bins]`.
    pub fn magnitude(&self) -> Vec<f32> {
        self.re
            .iter()
            .zip(&self.im)
            .map(|(r, i)| (r * r + i * i).sqrt())
            .collect()
    }
}

/// Computes the STFT of a real `signal` under `attrs`.
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`] if `n_fft == 0`, `hop_length == 0`,
/// or `win_length` is `0` or greater than `n_fft`.
pub fn stft(signal: &[f32], attrs: &StftAttrs) -> Result<Spectrogram> {
    if attrs.n_fft == 0 || attrs.hop_length == 0 {
        return Err(VokraError::InvalidArgument(
            "stft: n_fft and hop_length must be non-zero".to_owned(),
        ));
    }
    if attrs.win_length == 0 || attrs.win_length > attrs.n_fft {
        return Err(VokraError::InvalidArgument(
            "stft: win_length must be in 1..=n_fft".to_owned(),
        ));
    }

    let n = attrs.n_fft;
    let hop = attrs.hop_length;
    let frame_window = build_frame_window(attrs);
    let padded = pad_for_analysis(signal, attrs);

    let frames = if padded.len() >= n {
        (padded.len() - n) / hop + 1
    } else {
        0
    };
    let bins = if attrs.real_input { n / 2 + 1 } else { n };
    let scale = norm_scale(attrs.normalization, n, true);

    let mut re = vec![0.0f32; frames * bins];
    let mut im = vec![0.0f32; frames * bins];

    // Reuse one plan across all frames (hot-path allocation is FR-EX-05 / M1).
    let real_plan = attrs.real_input.then(|| RealFftPlan::new(n));
    let complex_plan = (!attrs.real_input).then(|| FftPlan::new(n));
    let mut frame = vec![0.0f32; n];

    for f in 0..frames {
        let start = f * hop;
        for i in 0..n {
            frame[i] = padded[start + i] * frame_window[i];
        }
        let base = f * bins;
        if let Some(plan) = &real_plan {
            let spec = plan.forward(&frame);
            write_bins(
                &spec,
                scale,
                &mut re[base..base + bins],
                &mut im[base..base + bins],
            );
        } else {
            let cin: Vec<Complex32> = frame.iter().map(|&s| Complex32::from_real(s)).collect();
            let spec = complex_plan
                .as_ref()
                .expect("complex plan")
                .forward_raw(&cin);
            write_bins(
                &spec,
                scale,
                &mut re[base..base + bins],
                &mut im[base..base + bins],
            );
        }
    }

    Ok(Spectrogram {
        frames,
        bins,
        re,
        im,
    })
}

fn write_bins(spec: &[Complex32], scale: f32, re: &mut [f32], im: &mut [f32]) {
    for (dst, c) in re.iter_mut().zip(spec) {
        *dst = c.re * scale;
    }
    for (dst, c) in im.iter_mut().zip(spec) {
        *dst = c.im * scale;
    }
}

/// Builds the length-`n_fft` analysis window, centering a shorter `win_length`
/// window and zero-padding it.
fn build_frame_window(attrs: &StftAttrs) -> Vec<f32> {
    let w = window(attrs.window, attrs.win_length, attrs.window_symmetry);
    if attrs.win_length == attrs.n_fft {
        return w;
    }
    let mut full = vec![0.0f32; attrs.n_fft];
    let offset = (attrs.n_fft - attrs.win_length) / 2;
    full[offset..offset + attrs.win_length].copy_from_slice(&w);
    full
}

/// Applies the analysis-time padding: causal (left history only), centered
/// (both ends by `n_fft/2`), or none.
fn pad_for_analysis(signal: &[f32], attrs: &StftAttrs) -> Vec<f32> {
    if attrs.causal {
        // No look-ahead: pad only the history side by (n_fft - hop).
        let left = attrs.n_fft.saturating_sub(attrs.hop_length);
        pad_signal(signal, left, 0, attrs.pad_mode)
    } else if attrs.center {
        let p = attrs.n_fft / 2;
        pad_signal(signal, p, p, attrs.pad_mode)
    } else {
        signal.to_vec()
    }
}

/// Pads `signal` by `left` / `right` samples under `mode`.
fn pad_signal(signal: &[f32], left: usize, right: usize, mode: PadMode) -> Vec<f32> {
    let n = signal.len();
    let mut out = Vec::with_capacity(left + n + right);
    for j in 0..left {
        let off = -((left - j) as isize);
        out.push(sample_at(signal, off, mode));
    }
    out.extend_from_slice(signal);
    for q in 1..=right {
        let off = (n as isize - 1) + q as isize;
        out.push(sample_at(signal, off, mode));
    }
    out
}

/// Reads `signal` at a possibly out-of-range index under the padding `mode`.
fn sample_at(signal: &[f32], off: isize, mode: PadMode) -> f32 {
    let n = signal.len() as isize;
    if n == 0 {
        return 0.0;
    }
    if (0..n).contains(&off) {
        return signal[off as usize];
    }
    match mode {
        PadMode::Constant => 0.0,
        PadMode::Edge => {
            if off < 0 {
                signal[0]
            } else {
                signal[(n - 1) as usize]
            }
        }
        PadMode::Reflect => signal[reflect_index(off, signal.len())],
    }
}

/// Mirrors `i` into `0..n` with period `2(n-1)` — numpy `mode="reflect"` (the
/// boundary sample is not repeated).
fn reflect_index(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let mut m = ((i % period) + period) % period;
    if m >= n as isize {
        m = period - m;
    }
    m as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::ir::graph::WindowSymmetry;

    #[test]
    fn reflect_padding_mirrors_without_repeating_edge() {
        // signal [10,20,30]; reflect index -1 -> 20, -2 -> 30, 3 -> 20.
        let s = [10.0f32, 20.0, 30.0];
        assert_eq!(sample_at(&s, -1, PadMode::Reflect), 20.0);
        assert_eq!(sample_at(&s, -2, PadMode::Reflect), 30.0);
        assert_eq!(sample_at(&s, 3, PadMode::Reflect), 20.0);
        assert_eq!(sample_at(&s, 4, PadMode::Reflect), 10.0);
    }

    #[test]
    fn edge_and_constant_padding_at_out_of_range_indices() {
        // signal [10,20,30], n=3.
        let s = [10.0f32, 20.0, 30.0];
        // Edge: clamp to the nearest boundary sample.
        assert_eq!(sample_at(&s, -1, PadMode::Edge), 10.0);
        assert_eq!(sample_at(&s, -2, PadMode::Edge), 10.0);
        assert_eq!(sample_at(&s, 3, PadMode::Edge), 30.0);
        assert_eq!(sample_at(&s, 4, PadMode::Edge), 30.0);
        // Constant: zeros outside the support.
        assert_eq!(sample_at(&s, -1, PadMode::Constant), 0.0);
        assert_eq!(sample_at(&s, 3, PadMode::Constant), 0.0);
        // In-range indices are returned verbatim for either mode.
        for (i, &v) in s.iter().enumerate() {
            assert_eq!(sample_at(&s, i as isize, PadMode::Edge), v);
            assert_eq!(sample_at(&s, i as isize, PadMode::Constant), v);
        }
    }

    #[test]
    fn stft_rejects_degenerate_sizes() {
        // The documented `# Errors` contract: n_fft/hop_length/win_length guards.
        let signal = vec![0.0f32; 800];
        let base = StftAttrs::new(400, 160);

        let mut a = base.clone();
        a.n_fft = 0;
        assert!(matches!(
            stft(&signal, &a),
            Err(VokraError::InvalidArgument(_))
        ));

        let mut a = base.clone();
        a.hop_length = 0;
        assert!(matches!(
            stft(&signal, &a),
            Err(VokraError::InvalidArgument(_))
        ));

        let mut a = base.clone();
        a.win_length = 0;
        assert!(matches!(
            stft(&signal, &a),
            Err(VokraError::InvalidArgument(_))
        ));

        let mut a = base.clone();
        a.win_length = a.n_fft + 1;
        assert!(matches!(
            stft(&signal, &a),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn frame_count_matches_torch_center_formula() {
        // len=16000, n_fft=400, hop=160, center: padded=16400, frames=101.
        let signal = vec![0.0f32; 16000];
        let attrs = StftAttrs::new(400, 160);
        let spec = stft(&signal, &attrs).unwrap();
        assert_eq!(spec.frames, 101);
        assert_eq!(spec.bins, 201);
    }

    #[test]
    fn dc_signal_energy_concentrates_at_bin0() {
        // A constant signal windowed by (periodic) Hann has its DFT equal to
        // the Hann spectrum: bin 0 dominates at the window sum (= N/2), the
        // main lobe leaks into bin 1, and bins >= 2 are negligible.
        let signal = vec![1.0f32; 2048];
        let attrs = StftAttrs::new(256, 128); // Hann / periodic by default
        let spec = stft(&signal, &attrs).unwrap();
        let f = spec.frames / 2; // interior frame
        let base = f * spec.bins;
        let mags: Vec<f32> = (0..spec.bins)
            .map(|b| spec.re[base + b].hypot(spec.im[base + b]))
            .collect();
        // Periodic Hann of length 256 sums to 128.
        assert!((mags[0] - 128.0).abs() < 1.0, "bin0 {}", mags[0]);
        let argmax = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(argmax, 0, "DC bin should dominate");
        // Beyond the Hann main lobe (bin >= 2) energy is negligible.
        let tail: f32 = mags[2..].iter().sum();
        assert!(tail < 1e-2, "tail {tail}");
    }

    #[test]
    fn single_cosine_peaks_at_expected_bin() {
        // Cosine at bin k0 => energy concentrated at bin k0 (rectangular-ish
        // long window, no center pad, integer periods per frame).
        let n_fft = 64;
        let k0 = 8;
        let signal: Vec<f32> = (0..n_fft * 4)
            .map(|t| {
                (2.0 * std::f64::consts::PI * k0 as f64 * t as f64 / n_fft as f64).cos() as f32
            })
            .collect();
        let mut attrs = StftAttrs::new(n_fft, n_fft);
        attrs.center = false;
        attrs.window = vokra_core::ir::graph::Window::Hann;
        let spec = stft(&signal, &attrs).unwrap();
        let base = 0;
        let mut argmax = 0;
        let mut best = -1.0f32;
        for b in 0..spec.bins {
            let mag = spec.re[base + b].hypot(spec.im[base + b]);
            if mag > best {
                best = mag;
                argmax = b;
            }
        }
        assert_eq!(argmax, k0);
    }

    #[test]
    fn causal_mode_has_no_look_ahead() {
        // Truncating the signal after sample S must not change frames whose
        // last-needed sample <= S. Compare a prefix STFT against the full one.
        let full: Vec<f32> = (0..2000).map(|t| (t as f32 * 0.03).sin()).collect();
        let mut attrs = StftAttrs::new(256, 128);
        attrs.causal = true;
        attrs.center = false;
        attrs.window_symmetry = WindowSymmetry::Periodic;
        let full_spec = stft(&full, &attrs).unwrap();
        let prefix = &full[..1000];
        let prefix_spec = stft(prefix, &attrs).unwrap();
        // Frames present in both must be bit-identical (no future dependence).
        let common = prefix_spec.frames.min(full_spec.frames);
        assert!(common > 2);
        for f in 0..common {
            for b in 0..full_spec.bins {
                let i = f * full_spec.bins + b;
                assert!((full_spec.re[i] - prefix_spec.re[i]).abs() < 1e-6);
                assert!((full_spec.im[i] - prefix_spec.im[i]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn win_length_shorter_than_n_fft_is_centered() {
        let mut attrs = StftAttrs::new(512, 128);
        attrs.win_length = 400;
        let w = build_frame_window(&attrs);
        assert_eq!(w.len(), 512);
        // Zero-padded shoulders, support in the middle.
        assert_eq!(w[0], 0.0);
        assert_eq!(w[511], 0.0);
        assert!(w[256] > 0.5);
    }

    #[test]
    fn full_vs_real_spectrum_agree_on_shared_bins() {
        let signal: Vec<f32> = (0..1024).map(|t| (t as f32 * 0.05).sin()).collect();
        let mut a_real = StftAttrs::new(256, 128);
        a_real.real_input = true;
        let mut a_full = a_real.clone();
        a_full.real_input = false;
        let s_real = stft(&signal, &a_real).unwrap();
        let s_full = stft(&signal, &a_full).unwrap();
        assert_eq!(s_real.bins, 129);
        assert_eq!(s_full.bins, 256);
        for f in 0..s_real.frames {
            for b in 0..s_real.bins {
                let ri = f * s_real.bins + b;
                let fi = f * s_full.bins + b;
                assert!((s_real.re[ri] - s_full.re[fi]).abs() < 1e-2);
                assert!((s_real.im[ri] - s_full.im[fi]).abs() < 1e-2);
            }
        }
    }
}
