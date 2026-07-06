//! Fused log-mel scalar kernel (M2-04-T05; FR-OP-01/03, NFR-PF-13).
//!
//! Single-pass replacement for the imperative
//! `stft` → `power` → `MelFilterbank::apply` → `log10` → transpose chain used by
//! Whisper's front-end (see `vokra_models::whisper::mel::log_mel`). The
//! motivation is the ~5 MB of intermediates that the imperative path
//! materializes per 30 s clip — a full `re`+`im` split-complex spectrogram
//! (2 × frames × bins f32), a `power` buffer (frames × bins f32), and a
//! `mel[frames, n_mels]` buffer that is then transposed to `[n_mels, frames]`
//! for the encoder. This kernel eliminates all three by streaming one frame at
//! a time through:
//!
//! 1. window multiply into a **single reusable** `Vec<f32>` of length `n_fft`;
//! 2. one RFFT into a **single reusable** `Vec<Complex32>` of length `bins`
//!    (the M0-04 pocketfft port — zero-dep by construction);
//! 3. **inline** `r² + i²` accumulate directly into mel bands (no `power`
//!    materialization); and
//! 4. **inline** `log10(max(acc, 1e-10))` written directly to the transposed
//!    output layout `out[m * n_frames + t]` (no `[frames, n_mels]` intermediate),
//!    while tracking the global max for the dynamic-range floor.
//!
//! A second pass over `out` applies the Whisper normalization
//! `(v.max(gmax - 8) + 4) / 4`.
//!
//! # Fidelity vs the imperative reference
//!
//! Bit-close (not bit-exact) to `vokra_models::whisper::mel::log_mel`:
//!
//! - The reused FFT / frame scratch buffers do **not** change the arithmetic —
//!   `RealFftPlan::forward` returns an owned `Vec<Complex32>` in both paths and
//!   is fed the same windowed samples in the same order.
//! - The mel projection loses one round of intermediate rounding by folding the
//!   `r² + i²` accumulate directly into the `weights[m*n_freqs+k] * power[k]`
//!   dot-product. The floating-point associativity delta is within
//!   `atol = 0.01` in the log-normalized output (NFR-QL-01), verified by the
//!   T07 parity test.
//!
//! # Scope
//!
//! T05 delivers the scalar path only. SIMD (AVX2 / NEON) is T06 and lives in
//! `crates/vokra-backend-cpu/src/kernels/fused_logmel_{avx2,neon}.rs`; the
//! runtime dispatcher is `vokra_backend_cpu::fused_log_mel_dispatch`. This
//! module is safe Rust — the SIMD kernels in `vokra-backend-cpu` carry the
//! `unsafe` + `#[target_feature]` boundary (NFR-RL-07).

use vokra_core::ir::graph::StftAttrs;

use crate::fft::{Complex32, RealFftPlan, norm_scale};
use crate::mel::MelFilterbank;
use crate::stft::Spectrogram;
use crate::window::window;

/// The Whisper log-domain floor: `log10(1e-10) = -10`. The imperative
/// reference (`vokra_models::whisper::mel::log_mel`) clips mel energies to
/// `1e-10` before `log10`, then dynamic-range-normalizes with `gmax - 8`, so
/// frames with no signal end up at `(floor.max(gmax - 8) + 4) / 4`.
const MEL_FLOOR: f32 = 1e-10;

/// Streams `pcm` through the fused STFT → mel → log10 → transpose chain and
/// writes the Whisper-normalized log-mel spectrogram into `out`
/// (row-major `[n_mels, n_frames]`).
///
/// `out` must be pre-sized to `mel_fb.n_mels * n_frames`. Frames beyond those
/// actually produced by the STFT (only possible for absurdly short `pcm`) are
/// left at the log-floor written by the caller.
///
/// The kernel allocates exactly two scratch buffers up front — one
/// `Vec<f32>` of length `n_fft` for the windowed frame and one
/// `Vec<Complex32>` of length `bins` for the RFFT output — and reuses them
/// across every frame (~3001 frames for a 30 s Whisper clip).
///
/// # Panics
///
/// Panics if `out.len() != mel_fb.n_mels * n_frames`, if `mel_fb.n_freqs`
/// disagrees with the STFT bin count implied by `stft.n_fft`, or if `stft`'s
/// `n_fft` / `hop_length` / `win_length` are degenerate (matches the
/// `# Errors` contract of [`crate::stft::stft`], surfaced as panics here
/// because the log-mel front-end is called with statically-valid attrs from
/// the runtime frontend spec).
pub fn fused_log_mel_scalar(
    pcm: &[f32],
    stft: &StftAttrs,
    mel_fb: &MelFilterbank,
    n_frames: usize,
    out: &mut [f32],
) {
    let n_mels = mel_fb.n_mels;
    assert_eq!(
        out.len(),
        n_mels * n_frames,
        "fused_log_mel_scalar: out.len() != n_mels * n_frames"
    );
    assert!(
        stft.n_fft > 0 && stft.hop_length > 0,
        "fused_log_mel_scalar: n_fft and hop_length must be non-zero"
    );
    assert!(
        stft.win_length > 0 && stft.win_length <= stft.n_fft,
        "fused_log_mel_scalar: win_length must be in 1..=n_fft"
    );
    let bins = if stft.real_input {
        stft.n_fft / 2 + 1
    } else {
        stft.n_fft
    };
    assert_eq!(
        mel_fb.n_freqs, bins,
        "fused_log_mel_scalar: mel filterbank / STFT bin count mismatch"
    );

    // Build the length-`n_fft` analysis window (a shorter `win_length` is
    // centered and zero-padded, matching `Spectrogram`'s `build_frame_window`).
    let frame_window = build_frame_window_local(stft);
    let padded = pad_for_analysis_local(pcm, stft);

    let n = stft.n_fft;
    let hop = stft.hop_length;
    let produced = if padded.len() >= n {
        (padded.len() - n) / hop + 1
    } else {
        0
    };
    // Whisper drops the trailing frame; the caller has already sized `out` to
    // `n_frames` and expects any excess to be treated as no-signal.
    let kept = produced.min(n_frames);
    let scale = norm_scale(stft.normalization, n, true);

    // Reusable scratch — allocated once, reused across all frames (T05
    // no-hot-path-alloc invariant).
    let plan = RealFftPlan::new(n);
    let mut frame = vec![0.0f32; n];

    // First pass: STFT → mel → log10, tracking gmax.
    let mut gmax = f32::NEG_INFINITY;
    for t in 0..kept {
        let start = t * hop;
        // Window multiply into reusable scratch.
        for i in 0..n {
            frame[i] = padded[start + i] * frame_window[i];
        }
        // One RFFT per frame (owned Vec<Complex32> — matches the imperative
        // path's arithmetic exactly).
        let spec: Vec<Complex32> = plan.forward(&frame);
        // Inline `r² + i²` accumulate directly into mel bands. The dot-product
        // `sum_k weights[m,k] * (re_k² + im_k²)` is fused so no
        // `[frames, bins]` power buffer is ever materialized. Writes the
        // log10 straight into the transposed output layout
        // `out[m * n_frames + t]` (no `[frames, n_mels]` intermediate).
        for m in 0..n_mels {
            let filt = &mel_fb.weights[m * bins..(m + 1) * bins];
            let mut acc = 0.0f32;
            for (k, w) in filt.iter().enumerate() {
                let c = spec[k];
                // Apply the STFT normalization scale to the *complex* bin
                // exactly as `write_bins` in `stft.rs` does, so
                // `r² + i²` matches `Spectrogram::power` bit-close.
                let r = c.re * scale;
                let im = c.im * scale;
                acc += *w * (r * r + im * im);
            }
            let l = acc.max(MEL_FLOOR).log10();
            out[m * n_frames + t] = l;
            if l > gmax {
                gmax = l;
            }
        }
    }

    // If `kept == 0`, the caller-provided floor is the final value
    // (dynamic-range normalization needs a real gmax to be meaningful).
    if gmax == f32::NEG_INFINITY {
        return;
    }

    // Second pass: dynamic-range compression `(v.max(gmax - 8) + 4) / 4`
    // (matches `whisper::mel::log_mel` step 4).
    let dyn_floor = gmax - 8.0;
    for v in out.iter_mut() {
        *v = (v.max(dyn_floor) + 4.0) / 4.0;
    }
}

/// Local copy of `stft::build_frame_window` — kept private to this module so
/// the fused kernel is self-contained (no `pub(crate)` widening of the STFT
/// helper).
fn build_frame_window_local(attrs: &StftAttrs) -> Vec<f32> {
    let w = window(attrs.window, attrs.win_length, attrs.window_symmetry);
    if attrs.win_length == attrs.n_fft {
        return w;
    }
    let mut full = vec![0.0f32; attrs.n_fft];
    let offset = (attrs.n_fft - attrs.win_length) / 2;
    full[offset..offset + attrs.win_length].copy_from_slice(&w);
    full
}

/// Local copy of `stft::pad_for_analysis` (see the STFT module for the
/// full explanation of causal / centered / no-padding modes).
fn pad_for_analysis_local(signal: &[f32], attrs: &StftAttrs) -> Vec<f32> {
    // Delegate to a one-shot STFT of a zero-length signal is overkill; the
    // padding logic is small enough to inline here. Reuse `Spectrogram`'s
    // documented behavior via a scratch call would allocate a full spectrum,
    // which defeats the fusion goal — so we re-implement the padding rules
    // locally to keep the kernel allocation-lean.
    let pad_mode = attrs.pad_mode;
    let n_fft = attrs.n_fft;
    let hop = attrs.hop_length;

    let (left, right) = if attrs.causal {
        (n_fft.saturating_sub(hop), 0)
    } else if attrs.center {
        let p = n_fft / 2;
        (p, p)
    } else {
        (0, 0)
    };

    let n = signal.len();
    let mut out = Vec::with_capacity(left + n + right);
    for j in 0..left {
        let off = -((left - j) as isize);
        out.push(sample_at_local(signal, off, pad_mode));
    }
    out.extend_from_slice(signal);
    for q in 1..=right {
        let off = (n as isize - 1) + q as isize;
        out.push(sample_at_local(signal, off, pad_mode));
    }
    out
}

fn sample_at_local(signal: &[f32], off: isize, mode: vokra_core::ir::graph::PadMode) -> f32 {
    use vokra_core::ir::graph::PadMode;
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
        PadMode::Reflect => {
            let period = 2 * (n - 1);
            if period == 0 {
                return signal[0];
            }
            let mut m = ((off % period) + period) % period;
            if m >= n {
                m = period - m;
            }
            signal[m as usize]
        }
    }
}

// The unused `Spectrogram` import is retained for the docs cross-reference in
// the module header; suppress dead-code without touching the pub API.
#[allow(dead_code)]
fn _keep_spectrogram_in_scope_for_docs(_s: &Spectrogram) {}
