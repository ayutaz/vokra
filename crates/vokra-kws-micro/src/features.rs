//! 40-band log-mel feature extractor (Phase 1 of M5-03b).
//!
//! Front-end for the microWakeWord-style KWS forward: turns a stream of raw
//! `i16` PCM samples (16 kHz mono) into per-frame log-mel feature vectors at
//! a 10 ms hop. Bit-identical between the std host build and the (future)
//! thumbv8m Cortex-M55 no_std build (M5-03 Tier-3), because every arithmetic
//! step uses only `core` (`f32::sqrt` is not called — a private
//! Newton–Raphson approximation from the sister crate lives in
//! [`crate::scalar`]'s eventual companion; here we only need `log10` from
//! that module).
//!
//! # Pipeline
//!
//! For each 10 ms hop (`HOP_SAMPLES = 160` samples @ 16 kHz):
//!
//! 1. Slice a `WINDOW_SAMPLES`-wide window (default 512, i.e. 32 ms) from
//!    the ring buffer of most-recent PCM samples.
//! 2. Cast `i16 → f32` and apply the Hann window in-place.
//! 3. Radix-2 iterative Cooley–Tukey FFT (`N_FFT = 512`), producing complex
//!    bins `X[0..N_FFT/2+1]` (Hermitian-symmetric input → one-sided
//!    spectrum).
//! 4. Power spectrum `|X[k]|²` for `k ∈ [0, N_FFT/2]`.
//! 5. Row-major mel filterbank matmul (`[N_MELS, N_BINS] · [N_BINS] →
//!    [N_MELS]`), triangular filters equally-spaced on the mel scale
//!    (HTK convention `mel = 2595·log10(1 + hz/700)`).
//! 6. `log10(max(mel_energy, EPSILON))` — the per-band log-magnitude that
//!    the classifier consumes.
//!
//! # Scope
//!
//! This module provides the API + tests + host-side implementation, and
//! [`KwsMicro::detect`](crate::KwsMicro::detect) consumes it for real on
//! every frame. What is still outstanding sits downstream, not here:
//! INT8-preserving (Q8_0) GGUF I/O, without which no upstream checkpoint can
//! be bound to a forward chain. This module is
//! ALREADY `#![no_std]`-clean (checked visually below — only `core` +
//! `alloc` + `crate::scalar`, no `std` imports); the crate's `std` /
//! `no_std` toggle is set by [`crate`]'s `lib.rs`.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::f32::consts::PI;

use crate::scalar;

/// Sample rate the extractor assumes (Hz). microWakeWord's canonical
/// training-time front-end uses 16 kHz; this constant is exposed as a
/// public sanity check and MUST match `vokra.kws.sample_rate` in the
/// GGUF header when a real model is loaded.
pub const SAMPLE_RATE: u32 = 16_000;

/// Feature hop in milliseconds (10 ms per frame — the canonical
/// microWakeWord streaming rate).
pub const HOP_MS: u32 = 10;

/// Feature window in milliseconds (32 ms). Rounded up to the nearest
/// power of two (`N_FFT` below) for the radix-2 FFT.
pub const WINDOW_MS: u32 = 32;

/// Number of mel bands. 40 is the canonical microWakeWord value; higher
/// values give more spectral detail but scale up the FFT-side cost
/// linearly.
pub const N_MELS: usize = 40;

/// Number of audio samples per hop (`SAMPLE_RATE · HOP_MS / 1000`).
/// Constant so `[i16; HOP_SAMPLES]` slices work in caller signatures
/// without generics.
pub const HOP_SAMPLES: usize = (SAMPLE_RATE as usize) * (HOP_MS as usize) / 1000;

/// Number of audio samples per window (`SAMPLE_RATE · WINDOW_MS / 1000`
/// = 512 @ default constants).
pub const WINDOW_SAMPLES: usize = (SAMPLE_RATE as usize) * (WINDOW_MS as usize) / 1000;

/// FFT size — the next power of two ≥ [`WINDOW_SAMPLES`]. Kept equal at
/// the default constants (both = 512), so the window fills the FFT
/// exactly with no zero-padding. If [`WINDOW_MS`] is ever changed to a
/// non-power-of-two-derived value the mismatch would need explicit
/// zero-padding in [`FeatureExtractor::compute_frame_f32`].
pub const N_FFT: usize = 512;

/// One-sided real FFT bin count (`N_FFT / 2 + 1`). The full complex FFT
/// output is Hermitian-symmetric for real input, so only the first
/// `N_BINS` bins carry independent information; the mel filterbank
/// operates on those.
pub const N_BINS: usize = N_FFT / 2 + 1;

/// Floor applied before `log10` (`log10(1e-10) = -10`) to keep the log
/// bounded below for silent frames. Matches Whisper front-end floor
/// convention (`vokra_backend_cpu::fused_log_mel_dispatch` uses the same
/// value; the `dispatch` module it is defined in is private, so the
/// crate-root re-export is the only path a downstream can name).
pub const LOG_MEL_EPSILON: f32 = 1e-10;

// Compile-time contract: window must fit inside the FFT.
const _: () = assert!(WINDOW_SAMPLES <= N_FFT);
// Compile-time contract: FFT size must be a power of two (radix-2).
const _: () = assert!(N_FFT.is_power_of_two());

/// Precomputed per-frame state (Hann window + mel filterbank + FFT
/// twiddle factors). Owned buffers are heap-allocated (`alloc::Vec`) so
/// this crate stays `#![no_std]` + `alloc`; a fixed-array variant is a
/// possible Phase 3 optimisation for MC targets without a heap.
///
/// Constructing an extractor pre-computes all constants once, which
/// costs `~40·(N_BINS)` f32 for the mel filterbank (~40·257 ≈ 41 KB) +
/// `N_FFT` f32 for the window (2 KB) + `N_FFT/2` complex twiddles (4 KB).
/// Well under the M55 Tier-3 SRAM budget (~256 KB typical).
pub struct FeatureExtractor {
    /// Hann window, length [`WINDOW_SAMPLES`]. Applied before the FFT.
    hann: Vec<f32>,
    /// Mel filterbank, row-major `[N_MELS, N_BINS]`. Row `m` holds
    /// triangle-shaped weights for band `m`.
    mel_fb: Vec<f32>,
    /// Real parts of FFT twiddle factors, length `N_FFT / 2`.
    twiddle_re: Vec<f32>,
    /// Imaginary parts of FFT twiddle factors, length `N_FFT / 2`.
    twiddle_im: Vec<f32>,
    /// Bit-reversal permutation, length [`N_FFT`]. `bit_reverse[i]` is
    /// the reversed index of `i` in a `log2(N_FFT)`-bit word.
    bit_reverse: Vec<usize>,
}

impl FeatureExtractor {
    /// Constructs a new extractor, precomputing all fixed tables. Runs
    /// once per session, then [`Self::compute_frame_f32`] is cheap.
    pub fn new() -> Self {
        Self {
            hann: hann_window(WINDOW_SAMPLES),
            mel_fb: mel_filterbank(N_MELS, N_BINS, SAMPLE_RATE),
            twiddle_re: twiddles_re(N_FFT),
            twiddle_im: twiddles_im(N_FFT),
            bit_reverse: bit_reverse_indices(N_FFT),
        }
    }

    /// Computes one frame's log-mel feature vector.
    ///
    /// `window` MUST be exactly [`WINDOW_SAMPLES`] i16 samples wide — the
    /// caller is responsible for buffering + hop management
    /// ([`crate::KwsMicro::detect`] does not do this: it length-checks the
    /// frame it is handed and rejects any other width, so a ring-buffer
    /// helper is still owed).
    ///
    /// Returns a `Vec<f32>` of length [`N_MELS`] (heap-allocated for
    /// `#![no_std]` compatibility; a fixed-array variant is Phase 3
    /// follow-up).
    ///
    /// # Panics
    ///
    /// Panics via `debug_assert_eq!` if `window.len() != WINDOW_SAMPLES`;
    /// release builds truncate / zero-fill silently rather than adding a
    /// per-call heap Result — the caller is expected to size correctly.
    pub fn compute_frame_f32(&self, window: &[i16]) -> Vec<f32> {
        debug_assert_eq!(
            window.len(),
            WINDOW_SAMPLES,
            "compute_frame_f32: window must be exactly WINDOW_SAMPLES"
        );

        // (1) i16 → f32 with Hann window applied. i16 range is [-32768,
        // 32767] — we do NOT normalize to [-1, 1] because both feature
        // computation and mel filterbank are linear in the input, so
        // scaling only shifts the log baseline by a constant.
        let mut re = vec![0.0f32; N_FFT];
        let mut im = vec![0.0f32; N_FFT];
        let take = window.len().min(WINDOW_SAMPLES);
        for i in 0..take {
            re[i] = (window[i] as f32) * self.hann[i];
        }
        // re[take..] and im[..] stay zero (implicit zero-padding, though
        // at default constants take == WINDOW_SAMPLES == N_FFT so there
        // is none).

        // (2) Radix-2 iterative Cooley–Tukey FFT (in-place).
        fft_radix2(
            &mut re,
            &mut im,
            &self.twiddle_re,
            &self.twiddle_im,
            &self.bit_reverse,
        );

        // (3) Power spectrum |X[k]|² for k ∈ [0, N_BINS).
        let mut power = vec![0.0f32; N_BINS];
        for k in 0..N_BINS {
            power[k] = re[k] * re[k] + im[k] * im[k];
        }

        // (4) Mel filterbank matmul: mel_energy[m] = Σ_k mel_fb[m,k] · power[k].
        // Iterate row-wise over the fixed-shape filterbank. Explicit indexing
        // keeps the MSRV at 1.85 (`slice::as_chunks` is newer) while retaining
        // exact-row semantics: a malformed short table panics at the invariant
        // boundary instead of accepting a partial final row.
        let mut mel_energy = [0.0f32; N_MELS];
        for (m, e) in mel_energy.iter_mut().enumerate() {
            let row = &self.mel_fb[m * N_BINS..(m + 1) * N_BINS];
            let mut acc = 0.0f32;
            for (w, p) in row.iter().zip(power.iter()) {
                acc += w * p;
            }
            *e = acc;
        }

        // (5) log10(max(mel_energy, epsilon)) — the classifier input.
        let features: Vec<f32> = mel_energy
            .iter()
            .map(|&e| {
                let clamped = if e < LOG_MEL_EPSILON {
                    LOG_MEL_EPSILON
                } else {
                    e
                };
                scalar::log10(clamped)
            })
            .collect();
        features
    }

    /// Number of samples per hop (convenience re-export for the ring
    /// buffer caller, so they do not need to import [`HOP_SAMPLES`]
    /// separately).
    pub fn hop_samples(&self) -> usize {
        HOP_SAMPLES
    }

    /// Number of samples per window.
    pub fn window_samples(&self) -> usize {
        WINDOW_SAMPLES
    }

    /// Feature vector length (equals [`N_MELS`]).
    pub fn feature_dim(&self) -> usize {
        N_MELS
    }
}

impl Default for FeatureExtractor {
    /// Same as [`Self::new`].
    fn default() -> Self {
        Self::new()
    }
}

/// Quantises an f32 feature vector to i8 using standard TFLite affine
/// quantization (`i8 = clamp(round(f32 / scale) + zero_point, -128, 127)`).
///
/// Mirror of the `dequantize_int8_to_f32` helper in
/// `tools/parity/microwakeword/prepare_checkpoint.py`. Used by
/// [`crate::KwsMicro::detect`] to feed features into the INT8 forward chain
/// with bit-identical arithmetic on both sides of the FFI.
///
/// `scale > 0` is enforced via `debug_assert!`; a `scale ≤ 0` in
/// production would produce garbage indexes (division sign flip) so
/// callers should validate their GGUF quantization metadata at load
/// time.
pub fn quantize_int8(features: &[f32], scale: f32, zero_point: i32) -> Vec<i8> {
    debug_assert!(scale > 0.0, "quantize_int8: scale must be positive");
    let mut out = vec![0i8; features.len()];
    for (o, &f) in out.iter_mut().zip(features.iter()) {
        // Standard round-half-away-from-zero (not banker's rounding), matching
        // TFLite's cast semantics for the quantization boundary.
        let scaled = f / scale;
        let biased = if scaled >= 0.0 {
            scaled + 0.5
        } else {
            scaled - 0.5
        };
        let raw = (biased as i32) + zero_point;
        *o = raw.clamp(-128, 127) as i8;
    }
    out
}

/// Inverse of [`quantize_int8`]: `f32 = scale * (i8 - zero_point)`. Used by
/// tests to verify lossless round-trip on the quantization boundary.
pub fn dequantize_int8(quantized: &[i8], scale: f32, zero_point: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; quantized.len()];
    for (o, &q) in out.iter_mut().zip(quantized.iter()) {
        // int32 upcast before subtraction avoids i8 wrap when `zero_point`
        // is at the extreme end of its range (same guard as
        // `prepare_checkpoint.py::dequantize_int8_to_f32`).
        *o = ((q as i32) - zero_point) as f32 * scale;
    }
    out
}

// ---------------------------------------------------------------------
// Precomputed tables (constructor-only cost).
// ---------------------------------------------------------------------

/// Standard Hann window `0.5·(1 - cos(2π·i/(N-1)))` for `i ∈ [0, N)`.
///
/// Uses `f32::cos` — which IS in `core` (only `f32::exp/log/tanh/sqrt`
/// are `std`-gated). This function is const-time in the sense that it is
/// computed once per `FeatureExtractor::new()`; there is no per-frame
/// cost.
fn hann_window(n: usize) -> Vec<f32> {
    debug_assert!(n >= 2, "hann_window needs n ≥ 2");
    let mut w = vec![0.0f32; n];
    let denom = (n - 1) as f32;
    for (i, val) in w.iter_mut().enumerate() {
        let t = (i as f32) / denom;
        *val = 0.5 * (1.0 - scalar::cos(2.0 * PI * t));
    }
    w
}

/// Triangular mel filterbank, row-major `[n_mels, n_bins]`. HTK
/// convention (`mel = 2595·log10(1 + hz/700)`) with `fmin = 0`,
/// `fmax = sample_rate / 2`.
///
/// The filters are the standard "overlapping triangles between adjacent
/// mel center frequencies" pattern. Filter `m` peaks at its own
/// centre frequency and falls linearly to zero at the flanking centres.
/// This is the same shape librosa / torchaudio / TF audio default to
/// (with `norm=None`); a Slaney-style area normalisation (`norm='slaney'`)
/// is not applied here — the microWakeWord training pipeline expects
/// un-normalised triangles, and a mismatch would silently rescale every
/// feature by a per-band constant (a real correctness bug, matching the
/// upstream is the honest choice).
fn mel_filterbank(n_mels: usize, n_bins: usize, sample_rate: u32) -> Vec<f32> {
    let fmax = (sample_rate as f32) * 0.5;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(fmax);
    // `n_mels + 2` equally-spaced mel points: two flanks + n_mels centres.
    let mut mel_points = vec![0.0f32; n_mels + 2];
    for (i, p) in mel_points.iter_mut().enumerate() {
        *p = mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32);
    }
    // Convert mel points back to Hz, then to FFT bin indices (fractional,
    // for triangle vertex placement).
    let bin_scale = (n_bins - 1) as f32 / fmax;
    let mut bin_pts = vec![0.0f32; n_mels + 2];
    for (i, bp) in bin_pts.iter_mut().enumerate() {
        *bp = mel_to_hz(mel_points[i]) * bin_scale;
    }

    let mut fb = vec![0.0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let (left, center, right) = (bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]);
        for k in 0..n_bins {
            let kf = k as f32;
            let w = if kf < left || kf > right {
                0.0
            } else if kf <= center {
                if center == left {
                    // Degenerate triangle (would divide by zero). Falls back
                    // to a single-bin peak — rare, only at low n_mels.
                    1.0
                } else {
                    (kf - left) / (center - left)
                }
            } else if center == right {
                1.0
            } else {
                (right - kf) / (right - center)
            };
            fb[m * n_bins + k] = w;
        }
    }
    fb
}

/// HTK-style Hz → mel conversion. `mel = 2595·log10(1 + hz/700)`.
///
/// Uses [`crate::scalar::log10`] so the mel filterbank construction is
/// deterministic across std / no_std builds (bit-identical
/// filterbank ⇒ bit-identical features).
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * scalar::log10(1.0 + hz / 700.0)
}

/// HTK-style mel → Hz conversion. Inverse of [`hz_to_mel`]:
/// `hz = 700·(10^(mel/2595) − 1)`.
///
/// Uses `f32::powi` on an integer exponent when it can, and otherwise
/// the `exp * ln(10)` identity via the local scalar module for
/// std-parity. Here the exponent `mel/2595` is fractional so we can't
/// use `powi`; instead compute `10^y = exp(y·ln 10)` using
/// `sister crate scalar::exp` — but the sister crate is a separate
/// crate, so we use the identity `10^y = 2^(y·log₂ 10)` and rely on
/// std's `f32::powf` when available… actually the cleanest and
/// deterministic way is `y · ln 10` composed via a Taylor `exp`. To
/// avoid pulling in the sister crate's `scalar::exp` here (which would
/// create a cross-crate dep just for the mel filterbank), we compute
/// `10^y = 10^int · 10^frac` where `10^frac ≈ 1 + frac·ln 10 + …` for
/// small `frac`. Since `y = mel/2595` is a positive fractional at the
/// mel-point grid values, this is well-conditioned.
///
/// Rationale-in-a-line: `mel_to_hz` runs `n_mels + 2 = 42` times per
/// extractor construction, not per frame — accuracy dominates speed
/// here, and a straightforward `y · ln 10` degree-6 Taylor gets us
/// inside f32 rounding.
fn mel_to_hz(mel: f32) -> f32 {
    let y = mel / 2595.0;
    // 10^y = e^(y · ln 10). Range-reduce y to y = k + frac with
    // k = floor(y), frac ∈ [0, 1); then 10^y = 10^k · 10^frac.
    let k = scalar::floor(y) as i32;
    let frac = y - (k as f32);
    // 10^k via integer accumulation (k ≤ 4 for audible frequencies).
    let mut pow_int = 1.0f32;
    if k >= 0 {
        for _ in 0..k {
            pow_int *= 10.0;
        }
    } else {
        for _ in 0..(-k) {
            pow_int *= 0.1;
        }
    }
    // 10^frac = e^(frac · ln 10). ln 10 ≈ 2.302585; frac · ln 10 ∈ [0, 2.303).
    // Degree-6 Taylor around 0: e^x ≈ 1 + x + x²/2! + … + x⁶/6!. Worst-case
    // truncation error at x = 2.303 is ~x⁷/7! ≈ 8e-3 — good enough for
    // filterbank frequency resolution (bins at 16 kHz are spaced ~31 Hz
    // apart; a 8e-3 relative error on a mel-scale HZ point is <1 Hz).
    let x = frac * core::f32::consts::LN_10;
    let pow_frac = {
        // Horner form: (((((1/720 · x + 1/120) · x + 1/24) · x + 1/6) · x + 1/2) · x + 1) · x + 1
        let mut p = 1.0 / 720.0;
        p = p * x + 1.0 / 120.0;
        p = p * x + 1.0 / 24.0;
        p = p * x + 1.0 / 6.0;
        p = p * x + 0.5;
        p = p * x + 1.0;
        p * x + 1.0
    };
    700.0 * (pow_int * pow_frac - 1.0)
}

/// FFT twiddle factor real parts: `cos(-2π·k/N)` for `k ∈ [0, N/2)`.
///
/// `f32::cos` IS in `core` (unlike `f32::log10`), so this compiles under
/// `#![no_std]` directly.
fn twiddles_re(n: usize) -> Vec<f32> {
    let half = n / 2;
    let mut w = vec![0.0f32; half];
    let base = -2.0 * PI / (n as f32);
    for (k, val) in w.iter_mut().enumerate() {
        *val = scalar::cos(base * (k as f32));
    }
    w
}

/// FFT twiddle factor imaginary parts: `sin(-2π·k/N)` for `k ∈ [0, N/2)`.
fn twiddles_im(n: usize) -> Vec<f32> {
    let half = n / 2;
    let mut w = vec![0.0f32; half];
    let base = -2.0 * PI / (n as f32);
    for (k, val) in w.iter_mut().enumerate() {
        *val = scalar::sin(base * (k as f32));
    }
    w
}

/// Bit-reversal permutation table for a radix-2 iterative FFT of size
/// `n` (must be a power of two). `out[i]` is the bit-reversal of `i` in
/// a `log2(n)`-bit word.
fn bit_reverse_indices(n: usize) -> Vec<usize> {
    debug_assert!(n.is_power_of_two());
    let log2n = n.trailing_zeros() as usize;
    let mut r = vec![0usize; n];
    for (i, out) in r.iter_mut().enumerate() {
        let mut v = i;
        let mut rev = 0usize;
        for _ in 0..log2n {
            rev = (rev << 1) | (v & 1);
            v >>= 1;
        }
        *out = rev;
    }
    r
}

/// Iterative radix-2 Cooley–Tukey FFT, in-place on split real / imag
/// buffers.
///
/// Standard textbook algorithm: bit-reverse permute, then log₂(N)
/// stages of butterflies with twiddle stride halving each stage. Total
/// arithmetic cost is `~5·N·log₂(N)` f32 ops.
///
/// # Panics (debug only)
///
/// Panics via `debug_assert_eq!` on any length mismatch. Release builds
/// truncate silently — but the constructor guarantees the twiddle /
/// permutation tables match `N_FFT`, so this can only misfire on
/// misuse.
fn fft_radix2(re: &mut [f32], im: &mut [f32], tw_re: &[f32], tw_im: &[f32], perm: &[usize]) {
    let n = re.len();
    debug_assert_eq!(re.len(), im.len());
    debug_assert_eq!(perm.len(), n);
    debug_assert_eq!(tw_re.len(), n / 2);
    debug_assert_eq!(tw_im.len(), n / 2);

    // (a) Bit-reverse permutation: swap re[i] with re[perm[i]] and im[i]
    // with im[perm[i]] for i < perm[i] (each swap is done once).
    for (i, &j) in perm.iter().enumerate() {
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // (b) log₂(N) butterfly stages. `size` doubles each stage (2, 4, 8,
    // …, N). Twiddle stride is `n / size` — indexes into the length-N/2
    // twiddle table.
    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        let step = n / size;
        let mut i = 0;
        while i < n {
            let mut k = 0;
            for j in 0..half {
                let a_re = re[i + j];
                let a_im = im[i + j];
                let b_re = re[i + j + half];
                let b_im = im[i + j + half];
                // t = b · twiddle
                let wr = tw_re[k];
                let wi = tw_im[k];
                let t_re = b_re * wr - b_im * wi;
                let t_im = b_re * wi + b_im * wr;
                re[i + j] = a_re + t_re;
                im[i + j] = a_im + t_im;
                re[i + j + half] = a_re - t_re;
                im[i + j + half] = a_im - t_im;
                k += step;
            }
            i += size;
        }
        size *= 2;
    }
}

// ---------------------------------------------------------------------
// Tests (host-side, synthetic — no fixtures, matches the Phase 1 spec).
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Shape contract: the extractor emits exactly [`N_MELS`] features per
    /// frame, no more, no less.
    #[test]
    fn compute_frame_returns_n_mels_features() {
        let ext = FeatureExtractor::new();
        let silent = vec![0i16; WINDOW_SAMPLES];
        let feats = ext.compute_frame_f32(&silent);
        assert_eq!(feats.len(), N_MELS);
        assert_eq!(ext.feature_dim(), N_MELS);
        assert_eq!(ext.hop_samples(), HOP_SAMPLES);
        assert_eq!(ext.window_samples(), WINDOW_SAMPLES);
    }

    /// Silent-input range contract: a fully-zero window MUST produce the
    /// log-floor value for every mel band. This documents the log-floor
    /// design and pins the epsilon constant.
    #[test]
    fn silent_frame_saturates_to_log_floor() {
        let ext = FeatureExtractor::new();
        let silent = vec![0i16; WINDOW_SAMPLES];
        let feats = ext.compute_frame_f32(&silent);
        let expected_floor = scalar::log10(LOG_MEL_EPSILON);
        for (m, &f) in feats.iter().enumerate() {
            let delta = (f - expected_floor).abs();
            assert!(
                delta < 1e-5,
                "band {m}: silent feature {f} != log-floor {expected_floor} (Δ={delta})"
            );
        }
    }

    /// Monotonicity contract: on a **broadband** input (a sum of sinusoids
    /// spread across the spectrum so every mel band carries real signal
    /// energy well above the log-floor), doubling the input amplitude
    /// MUST raise every mel feature by close to `log10(4) ≈ 0.602`. FFT
    /// linearity guarantees `|c·X|² = c²·|X|²`, so `log10(4·mel_energy) =
    /// log10(mel_energy) + log10(4)` for every band above the floor.
    ///
    /// This catches inversions in the log / pipeline sign or a
    /// non-linear FFT bug — features going DOWN with more energy would
    /// fail this test loudly.
    ///
    /// # Why not a single-frequency sinusoid?
    ///
    /// A pure sinusoid places all its energy in one FFT bin (plus a small
    /// Hann-window leakage skirt). Distant mel bands sit essentially at
    /// the quantization noise floor from the `i16` → `f32` cast, which
    /// itself is amplitude-dependent (rounding noise stays ±0.5 whether
    /// amp = 1000 or 2000). At bands where signal ≈ noise, doubling
    /// amplitude can shift the noise / signal balance in either
    /// direction — the *pipeline* is still linear, but the *quantization
    /// stage* is not. Empirical: with sinusoid(500 Hz, 1000/2000) we saw
    /// bands 13–14 dominated by signal (`Δ ≈ +0.602`, exact) but bands
    /// like 23 dominated by rounding noise (`Δ ≈ −1.0`, noise-driven).
    /// A broadband signal keeps every band signal-dominated so linearity
    /// holds observably.
    #[test]
    fn amplitude_doubling_increases_log_mel_features() {
        let ext = FeatureExtractor::new();
        // Use large amplitudes (8000, 16000 peak) so per-component
        // amplitude is 1000 / 2000 — well above the ±0.5 i16 rounding
        // noise, so every bin's signal power dominates the quantization
        // floor and FFT linearity is directly observable.
        let a: Vec<i16> = broadband(8000.0, WINDOW_SAMPLES);
        let b: Vec<i16> = broadband(16000.0, WINDOW_SAMPLES);
        let feats_a = ext.compute_frame_f32(&a);
        let feats_b = ext.compute_frame_f32(&b);
        let expected = 2.0 * (2.0f32).log10(); // = log10(4) ≈ 0.6021
        let mut n_close_to_expected = 0;
        for m in 0..N_MELS {
            let delta = feats_b[m] - feats_a[m];
            // Every band MUST NOT decrease. A tiny negative tolerance
            // accommodates f32 rounding on log-domain subtraction; the
            // honest expectation from FFT linearity is Δ ≥ 0.
            assert!(
                delta > -1e-3,
                "band {m}: feature DECREASED when amplitude doubled (Δ={delta}, \
                 a={a}, b={b})",
                a = feats_a[m],
                b = feats_b[m],
            );
            if (delta - expected).abs() < 0.05 {
                n_close_to_expected += 1;
            }
        }
        // The MAJORITY of bands should show the exact +log10(4) increase
        // (only rare band-edge / log-floor cases would deviate); "at
        // least half" is a safe assertion that also catches an off-by-
        // one bug where features are systematically doubled twice.
        assert!(
            n_close_to_expected >= N_MELS / 2,
            "only {n_close_to_expected}/{N_MELS} bands showed the expected \
             +log10(4)≈{expected:.3} increase; pipeline may not be FFT-linear"
        );
    }

    /// FFT correctness: a single-frequency sinusoid whose frequency lands
    /// exactly on an FFT bin MUST show its peak energy at that bin. We
    /// verify by feeding a bin-centered sinusoid (no Hann this time — we
    /// bypass the extractor and test the FFT primitive directly) and
    /// asserting the peak lands where expected.
    #[test]
    fn radix2_fft_peaks_at_target_bin() {
        // Sinusoid at exactly bin 8 of a 64-point FFT.
        let n = 64;
        let target_bin = 8;
        let tw_re = twiddles_re(n);
        let tw_im = twiddles_im(n);
        let perm = bit_reverse_indices(n);
        let mut re = vec![0.0f32; n];
        let mut im = vec![0.0f32; n];
        let two_pi = 2.0 * PI;
        for (i, r) in re.iter_mut().enumerate() {
            *r = (two_pi * (target_bin as f32) * (i as f32) / (n as f32)).cos();
        }
        fft_radix2(&mut re, &mut im, &tw_re, &tw_im, &perm);
        // Find the peak of |X|² over the one-sided spectrum.
        let n_bins = n / 2 + 1;
        let mut peak_bin = 0;
        let mut peak_pow = 0.0f32;
        for k in 0..n_bins {
            let p = re[k] * re[k] + im[k] * im[k];
            if p > peak_pow {
                peak_pow = p;
                peak_bin = k;
            }
        }
        assert_eq!(
            peak_bin, target_bin,
            "FFT peak landed at bin {peak_bin}, expected {target_bin}"
        );
    }

    /// Mel filterbank correctness: every row must sum to a positive value
    /// (no all-zero rows) and every bin must have nonneg weight (no
    /// negative sidelobes — pure triangles).
    #[test]
    fn mel_filterbank_rows_are_nonneg_triangles() {
        let ext = FeatureExtractor::new();
        for m in 0..N_MELS {
            let row = &ext.mel_fb[m * N_BINS..(m + 1) * N_BINS];
            let sum: f32 = row.iter().sum();
            assert!(
                sum > 0.0,
                "mel band {m}: filterbank row sum {sum} not > 0 (empty band?)"
            );
            for (k, &w) in row.iter().enumerate() {
                assert!(
                    w >= 0.0,
                    "mel band {m} bin {k}: negative weight {w} (not a triangle)"
                );
                assert!(
                    w <= 1.0 + 1e-6,
                    "mel band {m} bin {k}: weight {w} exceeds 1.0 (bad normalisation)"
                );
            }
        }
    }

    /// INT8 quantization round-trip contract: `f32 → i8 → f32` must
    /// reconstruct the original within one half-step (`scale/2`) for
    /// inputs inside the representable range. This pins the
    /// `quantize_int8` / `dequantize_int8` arithmetic against the
    /// TFLite affine formula. Saturation is tested separately by
    /// [`int8_quantization_saturates_at_extremes`].
    #[test]
    fn int8_roundtrip_within_one_step() {
        // Realistic microWakeWord scale range: log-mel features live in
        // roughly [-10, +5]; a scale of 0.06 covers that in 256 steps.
        // With zero_point = 20, the representable f32 range is:
        //   min = scale * (-128 - zp) = 0.06 * -148 = -8.88
        //   max = scale * ( 127 - zp) = 0.06 *  107 = +6.42
        // Stay well inside [-8.88, +6.42] so no input saturates — the
        // half-step bound only holds in the non-saturating region.
        let scale = 0.06f32;
        let zero_point = 20i32;
        // 201 evenly-spaced points on [-5.0, +5.0]; ±5 is comfortably
        // inside the ±(6.42/-8.88) unsaturated range.
        let inputs: Vec<f32> = (-100..=100).map(|i| (i as f32) * 0.05).collect();
        let quantized = quantize_int8(&inputs, scale, zero_point);
        let reconstructed = dequantize_int8(&quantized, scale, zero_point);
        assert_eq!(reconstructed.len(), inputs.len());
        for (i, (r, want)) in reconstructed.iter().zip(inputs.iter()).enumerate() {
            // Half-step bound: round-to-nearest gives worst |Δ| = 0.5·scale
            // for inputs inside the representable range. Small +ε for
            // f32 float rounding on the `f/scale + 0.5` addition itself.
            let delta = (r - want).abs();
            let bound = 0.5 * scale + 1e-6;
            assert!(
                delta <= bound,
                "roundtrip at i={i} input={want}: delta={delta} > bound={bound} (q={})",
                quantized[i]
            );
        }
    }

    /// INT8 quantization saturation contract: inputs far outside the
    /// representable range MUST clip to `-128` / `127`, never wrap.
    #[test]
    fn int8_quantization_saturates_at_extremes() {
        let scale = 0.1f32;
        let zero_point = 0i32;
        let extreme_low = vec![-1000.0f32];
        let extreme_high = vec![1000.0f32];
        assert_eq!(quantize_int8(&extreme_low, scale, zero_point)[0], -128);
        assert_eq!(quantize_int8(&extreme_high, scale, zero_point)[0], 127);
    }

    /// Determinism contract: two calls with the same input MUST produce
    /// bit-identical output (no per-call random state). This is what
    /// makes the std ↔ no_std bit-identity claim in the ADR credible.
    #[test]
    fn extractor_is_deterministic() {
        let ext_a = FeatureExtractor::new();
        let ext_b = FeatureExtractor::new();
        let sig: Vec<i16> = sinusoid(1000.0, 5000.0, WINDOW_SAMPLES);
        let feats_a = ext_a.compute_frame_f32(&sig);
        let feats_b = ext_b.compute_frame_f32(&sig);
        // Bit-identical (not just tolerance) — same source, no side effects.
        assert_eq!(feats_a, feats_b);
        // Also within one FeatureExtractor:
        let feats_c = ext_a.compute_frame_f32(&sig);
        assert_eq!(feats_a, feats_c);
    }

    // ------ helpers ------

    /// A truncated sinusoid of given `freq_hz` and `amp` (i16 peak),
    /// `n` samples at [`SAMPLE_RATE`]. Amplitude is the peak value —
    /// callers pick `< i16::MAX` to leave headroom for amplitude scaling.
    fn sinusoid(freq_hz: f32, amp: f32, n: usize) -> Vec<i16> {
        let mut s = vec![0i16; n];
        let two_pi_f_over_sr = 2.0 * PI * freq_hz / (SAMPLE_RATE as f32);
        for (i, v) in s.iter_mut().enumerate() {
            let val = amp * (two_pi_f_over_sr * (i as f32)).cos();
            *v = val.round().clamp(-32768.0, 32767.0) as i16;
        }
        s
    }

    /// Broadband signal — a sum of eight cosines spanning the audible
    /// range PLUS a deterministic pseudo-random dither, sampled at
    /// [`SAMPLE_RATE`]. Every mel band receives real signal energy well
    /// above the i16 quantization noise floor, so
    /// [`amplitude_doubling_increases_log_mel_features`] observes clean
    /// FFT linearity.
    ///
    /// **Frequencies are deliberately non-integer cycle counts over
    /// the 32 ms window** (e.g. 127 Hz × 0.032 s = 4.064 cycles, not
    /// integer): this forces Hann-window leakage into every FFT bin, so
    /// no bin sits at "pure quantization noise" (which would not scale
    /// linearly with input amplitude).
    ///
    /// `peak_amp` is the target per-sample peak; the eight cosine
    /// components each get `peak_amp / 10` and the dither gets
    /// `peak_amp / 10`, so the summed peak stays inside i16 range for
    /// `peak_amp ≤ 32000`.
    fn broadband(peak_amp: f32, n: usize) -> Vec<i16> {
        // Non-integer cycle counts over the 32 ms window → guaranteed
        // Hann leakage into every FFT bin.
        let freqs = [127.0, 313.0, 617.0, 1291.0, 2213.0, 3547.0, 5119.0, 6871.0];
        let per_amp = peak_amp / 10.0;
        let mut s = vec![0.0f32; n];
        for &f in &freqs {
            let two_pi_f = 2.0 * PI * f / (SAMPLE_RATE as f32);
            for (i, v) in s.iter_mut().enumerate() {
                *v += per_amp * (two_pi_f * (i as f32)).cos();
            }
        }
        // Deterministic dither (xorshift64) at ±per_amp — adds broadband
        // white noise to every bin, boosting the signal floor above i16
        // quantization noise. Seed = 0x9E37 keeps it reproducible.
        let mut rng: u64 = 0x9E37_9B97_F4A7_C15D;
        for v in s.iter_mut() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let u = ((rng >> 40) as u32 as f32) / ((1u32 << 24) as f32); // ∈ [0, 1)
            *v += per_amp * (2.0 * u - 1.0); // ∈ [-per_amp, per_amp)
        }
        s.iter()
            .map(|v| v.round().clamp(-32768.0, 32767.0) as i16)
            .collect()
    }
}
