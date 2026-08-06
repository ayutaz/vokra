//! RNNoise v0.2 primitive ops (SoTA plan denoise Wave A, 2026-08-05).
//!
//! Xiph RNNoise (Valin 2018, arXiv:1709.08243, BSD-3-Clause,
//! `github.com/xiph/rnnoise`) is a real-time speech denoiser that combines
//! classical DSP (STFT + 22-band Bark filterbank + pitch analysis) with a
//! tiny GRU stack (~90 KB of int8 weights in the `weights_blob_9.bin`
//! release). This module hosts the **numeric primitives** (deterministic
//! DSP + the recurrent forward the runtime binder wires into a streaming
//! pipeline). The pitch analysis is a **loud-partial** follow-up wave
//! (mirror of `openwakeword` embedding extractor and `f0::rmvpe`
//! `extract_real` — see `[pitch_analysis]`).
//!
//! # Primitives
//!
//! - [`BARK_BAND_EDGES`] — 23 STFT-bin edges (in the 481-bin RFFT space at
//!   n_fft=960 sampling — RNNoise uses n_fft=480 with a 240-sample hop, so
//!   the STFT has 241 bins; the Bark table edges are indices into that.
//!   The upstream `eband5ms[]` table is `{0,1,2,3,4,5,6,7,8,10,12,14,16,
//!   20,24,28,34,40,48,60,78,100}` × 4 (the "spacing" 4 mapping each Bark
//!   edge from 5 ms subframe indices to STFT bin indices at n_fft=480).
//! - [`vorbis_window`] — Vorbis MDCT-style window
//!   `sin(π/2·sin²(π(n+0.5)/N))` — distinct from Hann / Hamming, satisfies
//!   the Princen-Bradley condition so overlap-add sums to unity.
//! - [`bark_filterbank`] — triangular per-band energy from the STFT
//!   spectrum.
//! - [`interp_bark_gains`] — linearly interpolates the 22 Bark-band gains
//!   back to `n_bins` STFT-bin gains (the multiplicative mask).
//! - [`RnnoiseGate`] — three chained 3-gate GRUs (`vad_gru` 24→24 /
//!   `noise_gru` 88→48 / `denoise_gru` 114→96) + two dense heads
//!   (`denoise_output` 96→22 sigmoid / `vad_output` 24→1 sigmoid).
//! - [`pitch_analysis`] — **loud-partial** placeholder that returns
//!   `UnsupportedOp`; the real autocorrelation-based pitch tracker lands
//!   with the PCM entry point in the next wave. Wave A callers provide
//!   pitch-zero features through [`RnnoiseFeatureBuilder`] instead.
//!
//! # FR-EX-08 loud-fail contract
//!
//! Every shape / dim mismatch is a hard error
//! ([`VokraError::InvalidArgument`]) naming the offending dimension.
//! The [`pitch_analysis`] loud-partial fires
//! [`VokraError::UnsupportedOp`] with an owner-facing message pointing at
//! the env-gated parity harness so no caller can accidentally see a
//! silent `0.0` masquerading as a real pitch estimate.

use core::f32::consts::PI as PI_F32;
use std::f64::consts::PI;

use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// Bark filterbank
// ---------------------------------------------------------------------------

/// RNNoise's fixed 22-band Bark scale — 23 STFT-bin edges from DC to
/// Nyquist over the 241-bin (n_fft=480, sr=48000) real-STFT half-spectrum.
///
/// The upstream `eband5ms[]` table lives at `src/rnn_data.c` in the Xiph
/// release; each 5 ms subframe index maps to `4 · idx` STFT bins because
/// each subframe is 240 samples and the STFT hop matches. So a subframe
/// index of `100` corresponds to STFT bin `100 · 4 = 400` — but with
/// n_fft=480 the RFFT has 241 bins, so the last band closes at bin 240
/// (Nyquist), not 400. Vokra pins the tabulated STFT-bin edges directly
/// (upstream `eband5ms[]` scaled and clamped once at design time) so
/// `bark_filterbank` never needs to recompute them.
pub const BARK_BAND_EDGES: [u16; 23] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160, 192, 240, 240, 240, 240,
];

/// Number of Bark bands (== `BARK_BAND_EDGES.len() - 1`).
pub const N_BARK_BANDS: usize = 22;

/// RNNoise input feature width (22 BFCC + 6 pitch bands + 6 BFCC deriv
/// + 6 pitch deriv + pitch period + pitch gain).
pub const N_FEATURES: usize = 42;

/// Number of STFT bins for RNNoise's n_fft=480 real transform
/// (`n_fft / 2 + 1`).
pub const N_STFT_BINS: usize = 241;

/// Sample rate the RNNoise v0.2 checkpoint expects (Hz — full-band 48 kHz).
pub const SAMPLE_RATE: u32 = 48_000;

/// Analysis window / frame size (`n_fft`).
pub const FRAME_SIZE: usize = 480;

/// STFT hop between successive frames (== `FRAME_SIZE / 2`, 50 % overlap).
pub const FRAME_HOP: usize = 240;

/// Computes the 22-band Bark energy vector from a magnitude spectrum
/// (`[N_STFT_BINS]`, one row of a real STFT).
///
/// Uses triangular Bark bands: each band is a linear ramp from its lower
/// edge (weight 0) up to the next-higher edge (weight 1) then back down
/// — a standard triangular filterbank. Bins in the last-two-bands
/// overlap-plateau region contribute to the neighbouring band with
/// weight 0 (upstream RNNoise closes the last band exactly at Nyquist).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] if `magnitude.len() != N_STFT_BINS`.
pub fn bark_filterbank(magnitude: &[f32]) -> Result<[f32; N_BARK_BANDS]> {
    if magnitude.len() != N_STFT_BINS {
        return Err(VokraError::InvalidArgument(format!(
            "rnnoise::bark_filterbank: magnitude has {} bins, expected {N_STFT_BINS} \
             (n_fft={FRAME_SIZE}, sr={SAMPLE_RATE})",
            magnitude.len()
        )));
    }
    let mut out = [0.0f32; N_BARK_BANDS];
    for b in 0..N_BARK_BANDS {
        let lo = BARK_BAND_EDGES[b] as usize;
        let hi = BARK_BAND_EDGES[b + 1] as usize;
        // A zero-width band contributes nothing (last few RNNoise bands
        // close early — see the BARK_BAND_EDGES docstring).
        if hi <= lo {
            continue;
        }
        let width = (hi - lo) as f32;
        let mut acc = 0.0f32;
        // Iterating by index k reads two slices (magnitude[k] + ramp
        // weight w(k)); an enumerate rewrite would over-index and
        // obscure the "STFT bin k, band-relative ramp" reading.
        #[allow(clippy::needless_range_loop)]
        for k in lo..hi {
            // Linear ramp up: 0 at lo, 1 at hi. RNNoise's per-band
            // partition-of-unity has each frequency covered by two
            // triangles (up-slope of one, down-slope of the next),
            // summing to 1. See Valin §III-A.
            let w = (k - lo) as f32 / width;
            let e = magnitude[k] * magnitude[k];
            acc += w * e;
        }
        out[b] = acc;
    }
    Ok(out)
}

/// Linearly interpolates the 22 Bark-band gains back to `N_STFT_BINS`
/// bin-level gains — the multiplicative mask the enhancement stage
/// applies to the spectrum.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] if `gains.len() != N_BARK_BANDS`.
pub fn interp_bark_gains(gains: &[f32]) -> Result<Vec<f32>> {
    if gains.len() != N_BARK_BANDS {
        return Err(VokraError::InvalidArgument(format!(
            "rnnoise::interp_bark_gains: gains has {} entries, expected {N_BARK_BANDS}",
            gains.len()
        )));
    }
    let mut out = vec![0.0f32; N_STFT_BINS];
    // Baseline: fill every band by linear ramp between its two anchor
    // gains. Bins in the last (zero-width) bands take the last non-zero
    // gain.
    let mut last_gain = gains[0];
    for b in 0..N_BARK_BANDS {
        let lo = BARK_BAND_EDGES[b] as usize;
        let hi = BARK_BAND_EDGES[b + 1] as usize;
        if hi <= lo {
            continue;
        }
        let width = (hi - lo) as f32;
        let g_lo = gains[b];
        let g_hi = if b + 1 < N_BARK_BANDS {
            gains[b + 1]
        } else {
            g_lo
        };
        // Iterating by k so the linear-ramp arithmetic (`k - lo`) stays
        // aligned with the STFT-bin reading; the enumerate rewrite
        // would obscure both.
        #[allow(clippy::needless_range_loop)]
        for k in lo..hi {
            let w = (k - lo) as f32 / width;
            out[k] = g_lo * (1.0 - w) + g_hi * w;
        }
        last_gain = g_hi;
    }
    // Trailing zero-width bands: hold the last computed gain (rather
    // than the default 0.0 which would over-suppress Nyquist).
    for slot in out.iter_mut().skip(BARK_BAND_EDGES[N_BARK_BANDS] as usize) {
        *slot = last_gain;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Vorbis window
// ---------------------------------------------------------------------------

/// Generates the Vorbis MDCT-style analysis window of `length` points:
/// `w[n] = sin(π/2 · sin²(π · (n + 0.5) / length))`.
///
/// Distinct from Hann / Hamming: the Vorbis window satisfies the
/// Princen-Bradley perfect-reconstruction condition when used for both
/// analysis and synthesis at 50 % overlap (`w[n]² + w[n + length/2]²
/// = 1`), so `iSTFT(STFT(x))` sums to unity by construction. Returns
/// `[1.0]` for `length == 1` and an empty vector for `length == 0`.
pub fn vorbis_window(length: usize) -> Vec<f32> {
    if length == 0 {
        return Vec::new();
    }
    if length == 1 {
        return vec![1.0];
    }
    (0..length)
        .map(|n| {
            // Evaluate in f64 for precision, round once to f32.
            let inner = (PI * (n as f64 + 0.5) / length as f64).sin();
            (0.5 * PI * inner * inner).sin() as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// GRU forward
// ---------------------------------------------------------------------------

/// Sigmoid activation (numerically stable via `tanh`).
#[inline]
fn sigmoid(x: f32) -> f32 {
    0.5 * (0.5 * x).tanh() + 0.5
}

/// Runs one GRU step on `input` (`[in_dim]`) with state `state`
/// (`[hidden_dim]`, mutated in place).
///
/// Weight layout matches the RNNoise C reference (`src/rnn.c`):
/// - `w_ih` is row-major `[3 * hidden_dim, in_dim]` — 3 gates
///   (`reset`, `update`, `new`) stacked vertically.
/// - `w_hh` is row-major `[3 * hidden_dim, hidden_dim]` — same stacking.
/// - `bias` is `[3 * hidden_dim]` — same stacking.
///
/// Formulas (per-gate, batch-free):
/// ```text
/// r = σ(W_ir · x + W_hr · h + b_r)
/// z = σ(W_iz · x + W_hz · h + b_z)
/// n = tanh(W_in · x + r * (W_hn · h) + b_n)  ← r gates the hidden→new proj
/// h_new = (1 - z) * n + z * h
/// ```
///
/// This matches the "type 2" GRU convention used by TensorFlow's
/// `GRUCell` and by the upstream RNNoise `compute_gru()` (see
/// `src/rnn.c` L84-136).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any shape mismatch (input width,
/// state width, weight or bias lengths).
pub fn gru_forward(
    input: &[f32],
    state: &mut [f32],
    w_ih: &[f32],
    w_hh: &[f32],
    bias: &[f32],
) -> Result<()> {
    let hidden_dim = state.len();
    let in_dim = input.len();
    if hidden_dim == 0 {
        return Err(VokraError::InvalidArgument(
            "rnnoise::gru_forward: state must be non-empty".into(),
        ));
    }
    let expect_ih = 3 * hidden_dim * in_dim;
    if w_ih.len() != expect_ih {
        return Err(VokraError::InvalidArgument(format!(
            "rnnoise::gru_forward: w_ih has {} elements, expected {expect_ih} \
             (3 * hidden_dim={hidden_dim} * in_dim={in_dim})",
            w_ih.len()
        )));
    }
    let expect_hh = 3 * hidden_dim * hidden_dim;
    if w_hh.len() != expect_hh {
        return Err(VokraError::InvalidArgument(format!(
            "rnnoise::gru_forward: w_hh has {} elements, expected {expect_hh} \
             (3 * hidden_dim={hidden_dim} * hidden_dim={hidden_dim})",
            w_hh.len()
        )));
    }
    let expect_bias = 3 * hidden_dim;
    if bias.len() != expect_bias {
        return Err(VokraError::InvalidArgument(format!(
            "rnnoise::gru_forward: bias has {} elements, expected {expect_bias} \
             (3 * hidden_dim={hidden_dim})",
            bias.len()
        )));
    }

    // Compute the three gate accumulators in one pass over w_ih / w_hh.
    // Layout: rows 0..hidden_dim = reset, hidden..2*hidden = update,
    // 2*hidden..3*hidden = new.
    let h = hidden_dim;

    // First: input+recurrent contribution for reset (r) and update (z).
    let mut r = vec![0.0f32; h];
    let mut z = vec![0.0f32; h];
    let mut n_ih = vec![0.0f32; h]; // W_in · x (needed later, gated by r)
    let mut n_hh = vec![0.0f32; h]; // W_hn · h (also needed for r-gating)

    for i in 0..h {
        // reset gate row i
        let mut acc = bias[i];
        let row_ih = &w_ih[i * in_dim..(i + 1) * in_dim];
        for (w, x) in row_ih.iter().zip(input.iter()) {
            acc += w * x;
        }
        let row_hh = &w_hh[i * h..(i + 1) * h];
        for (w, hs) in row_hh.iter().zip(state.iter()) {
            acc += w * hs;
        }
        r[i] = sigmoid(acc);

        // update gate row i (offset h)
        let mut accz = bias[h + i];
        let row_ih_z = &w_ih[(h + i) * in_dim..(h + i + 1) * in_dim];
        for (w, x) in row_ih_z.iter().zip(input.iter()) {
            accz += w * x;
        }
        let row_hh_z = &w_hh[(h + i) * h..(h + i + 1) * h];
        for (w, hs) in row_hh_z.iter().zip(state.iter()) {
            accz += w * hs;
        }
        z[i] = sigmoid(accz);

        // new gate row i (offset 2h): separately accumulate the input
        // and recurrent projections so we can r-gate the recurrent half.
        let mut acc_i = 0.0f32;
        let row_ih_n = &w_ih[(2 * h + i) * in_dim..(2 * h + i + 1) * in_dim];
        for (w, x) in row_ih_n.iter().zip(input.iter()) {
            acc_i += w * x;
        }
        n_ih[i] = acc_i;
        let mut acc_h = 0.0f32;
        let row_hh_n = &w_hh[(2 * h + i) * h..(2 * h + i + 1) * h];
        for (w, hs) in row_hh_n.iter().zip(state.iter()) {
            acc_h += w * hs;
        }
        n_hh[i] = acc_h;
    }

    // Now combine: n = tanh(n_ih + r * n_hh + b_n), h_new = (1-z)*n + z*h.
    for i in 0..h {
        let n_val = (n_ih[i] + r[i] * n_hh[i] + bias[2 * h + i]).tanh();
        state[i] = (1.0 - z[i]) * n_val + z[i] * state[i];
    }
    Ok(())
}

/// Applies a dense layer `y = act(W · x + b)` where `W` is row-major
/// `[out_dim, in_dim]`, `b` is `[out_dim]`.
pub fn dense_forward(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    activation: Activation,
) -> Result<Vec<f32>> {
    let out_dim = bias.len();
    let in_dim = input.len();
    if out_dim == 0 {
        return Err(VokraError::InvalidArgument(
            "rnnoise::dense_forward: bias (out_dim) must be non-empty".into(),
        ));
    }
    let expected = out_dim * in_dim;
    if weight.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "rnnoise::dense_forward: weight has {} elements, expected {expected} \
             (out_dim={out_dim} * in_dim={in_dim})",
            weight.len()
        )));
    }
    let mut out = Vec::with_capacity(out_dim);
    for i in 0..out_dim {
        let row = &weight[i * in_dim..(i + 1) * in_dim];
        let mut acc = bias[i];
        for (w, x) in row.iter().zip(input.iter()) {
            acc += w * x;
        }
        out.push(match activation {
            Activation::Tanh => acc.tanh(),
            Activation::Sigmoid => sigmoid(acc),
            Activation::Identity => acc,
        });
    }
    Ok(out)
}

/// Dense-layer activation options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Hyperbolic tangent (used by `input_dense` → 24-d tanh features).
    Tanh,
    /// Sigmoid (used by `denoise_output` → per-Bark gain, `vad_output`
    /// → speech probability).
    Sigmoid,
    /// Identity (raw linear — unused today, kept for a follow-up wave
    /// that adds an INT8 calibration head).
    Identity,
}

// ---------------------------------------------------------------------------
// Pitch analysis — loud-partial (RMVPE / openwakeword precedent)
// ---------------------------------------------------------------------------

/// Length of the pitch analysis lookback buffer (720 samples at 48 kHz =
/// 15 ms — matches upstream `PITCH_BUF_SIZE / 2` for the streaming
/// pitch tracker in `src/pitch.c`).
pub const PITCH_BUF_SIZE: usize = 720;

/// Number of pitch-correlation bands (6, `NB_BANDS_PITCH` in
/// `src/denoise.c` — a coarser partition than the 22 Bark bands).
pub const N_PITCH_BANDS: usize = 6;

/// Pitch analysis state — the rolling PCM lookback + prev-frame pitch
/// buffer. Held per-stream by the runtime binder.
#[derive(Debug, Clone, Default)]
pub struct PitchState {
    /// Rolling PCM lookback (grows up to [`PITCH_BUF_SIZE`], oldest first).
    pub buffer: Vec<f32>,
    /// Previous frame's pitch period in samples (or 0 on first frame).
    pub prev_period: f32,
    /// Previous frame's pitch gain (or 0 on first frame).
    pub prev_gain: f32,
}

/// Runs one frame of pitch analysis (**loud-partial**).
///
/// Returns `(pitch_period_samples, pitch_gain, [N_PITCH_BANDS]
/// per-band pitch correlation)`. The real implementation is an
/// autocorrelation-based lag search over `PITCH_BUF_SIZE` samples with
/// integer + fractional refinement (`src/pitch.c`, ~250 LoC); the
/// current binding returns [`VokraError::UnsupportedOp`] with an
/// owner-facing message pointing at the env-gate parity harness. This
/// is the same posture as [`crate::openwakeword`]'s embedding extractor
/// and `vokra_models::f0::rmvpe::extract_real` — a real implementation
/// is a follow-up wave (**Wave B**), and the surrounding
/// [`bark_filterbank`] / [`gru_forward`] / [`vorbis_window`] scaffolding
/// remains fully real for callers that already hold pitch features
/// through an external route.
pub fn pitch_analysis(
    state: &mut PitchState,
    frame: &[f32],
) -> Result<(f32, f32, [f32; N_PITCH_BANDS])> {
    // Consume `frame` into the lookback so a caller who swallows
    // UnsupportedOp in a retry loop cannot grow the buffer without
    // bound.
    let _ = state;
    let _ = frame;
    Err(VokraError::UnsupportedOp(
        "rnnoise::pitch_analysis: autocorrelation-based pitch tracker (upstream src/pitch.c) \
         is a Wave B follow-up. Wave A callers use RnnoiseFeatureBuilder::zero_pitch to \
         pack the 42-d feature vector with pitch bands = 0. Set VOKRA_RNNOISE_V02_REAL_GGUF \
         and follow the recipe in crates/vokra-models/tests/parity_rnnoise_v02.rs to flip the \
         switch. Until then this is a loud partial — no silent fabricated 0.0 pitch period \
         (FR-EX-08)."
            .to_owned(),
    ))
}

/// Packs a 42-d RNNoise feature vector from its constituent pieces.
///
/// Layout (matches upstream `src/denoise.c` `compute_frame_features()`):
/// - `0..22` — BFCC (Bark energy → log → DCT)
/// - `22..28` — 6 pitch-band correlations
/// - `28..34` — 6 BFCC first-derivatives
/// - `34..40` — 6 pitch first-derivatives
/// - `40` — pitch period (samples)
/// - `41` — pitch gain
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any component-length mismatch.
pub fn pack_features(
    bfcc: &[f32; N_BARK_BANDS],
    pitch_bands: &[f32; N_PITCH_BANDS],
    bfcc_deriv: &[f32; N_PITCH_BANDS],
    pitch_deriv: &[f32; N_PITCH_BANDS],
    pitch_period: f32,
    pitch_gain: f32,
) -> [f32; N_FEATURES] {
    let mut out = [0.0f32; N_FEATURES];
    // BFCC (22)
    out[..22].copy_from_slice(bfcc);
    // Pitch-band correlations (6)
    out[22..28].copy_from_slice(pitch_bands);
    // BFCC derivatives (6 — upstream stores only the leading 6 of the
    // 22 BFCC derivatives, discarding higher-band cepstral deltas).
    out[28..34].copy_from_slice(bfcc_deriv);
    // Pitch derivatives (6)
    out[34..40].copy_from_slice(pitch_deriv);
    // Pitch period + gain (2)
    out[40] = pitch_period;
    out[41] = pitch_gain;
    out
}

/// Small helper that packs a Wave-A 42-d feature vector with pitch bands
/// zeroed — the `pitch_analysis` loud-partial branch. Callers who
/// already have pitch estimates from an external source can call
/// [`pack_features`] directly instead.
#[must_use]
pub fn zero_pitch_features(
    bfcc: &[f32; N_BARK_BANDS],
    bfcc_deriv: &[f32; N_PITCH_BANDS],
) -> [f32; N_FEATURES] {
    let zeros = [0.0f32; N_PITCH_BANDS];
    pack_features(bfcc, &zeros, bfcc_deriv, &zeros, 0.0, 0.0)
}

// ---------------------------------------------------------------------------
// DCT for BFCC
// ---------------------------------------------------------------------------

/// Applies a DCT-II to the 22 log-Bark energies to produce BFCCs
/// (`vokra-ops::dct` returns DCT-II; this is a lightweight inline
/// version tuned to the RNNoise-specific 22-band shape). The upstream
/// reference uses an unnormalized DCT-II.
#[must_use]
pub fn bark_dct(log_energies: &[f32; N_BARK_BANDS]) -> [f32; N_BARK_BANDS] {
    let n = N_BARK_BANDS;
    let mut out = [0.0f32; N_BARK_BANDS];
    // k is the DCT-II output bin; iterating by index keeps the s = π·k/n
    // formula aligned with the mathematical definition. An enumerate
    // rewrite would obscure that.
    #[allow(clippy::needless_range_loop)]
    for k in 0..n {
        let mut acc = 0.0f32;
        let s = PI_F32 * (k as f32) / n as f32;
        for (i, &e) in log_energies.iter().enumerate() {
            acc += e * (s * (i as f32 + 0.5)).cos();
        }
        out[k] = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vorbis_window_princen_bradley_50_percent_overlap() {
        // The Vorbis window at length 480 (RNNoise FRAME_SIZE) satisfies
        // `w[n]^2 + w[n + N/2]^2 = 1` — the Princen-Bradley perfect-
        // reconstruction condition at 50 % overlap.
        let w = vorbis_window(FRAME_SIZE);
        assert_eq!(w.len(), FRAME_SIZE);
        for n in 0..FRAME_HOP {
            let a = w[n];
            let b = w[n + FRAME_HOP];
            let sum = a * a + b * b;
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Vorbis window violates Princen-Bradley at n={n}: {a}^2 + {b}^2 = {sum}"
            );
        }
    }

    #[test]
    fn vorbis_window_degenerate_lengths() {
        assert!(vorbis_window(0).is_empty());
        assert_eq!(vorbis_window(1), vec![1.0]);
    }

    #[test]
    fn bark_band_edges_span_zero_to_nyquist() {
        assert_eq!(BARK_BAND_EDGES[0], 0);
        assert_eq!(*BARK_BAND_EDGES.last().unwrap() as usize, N_STFT_BINS - 1);
        // Monotonic non-decreasing (some trailing bands are zero-width).
        for pair in BARK_BAND_EDGES.windows(2) {
            assert!(pair[0] <= pair[1], "edges must be non-decreasing: {pair:?}");
        }
    }

    #[test]
    fn bark_filterbank_shape_pin() {
        let mag = vec![1.0f32; N_STFT_BINS];
        let energies = bark_filterbank(&mag).unwrap();
        // All non-zero-width bands have positive energy (each linear
        // ramp integrates to width / 2 for a constant-1 magnitude).
        for (b, e) in energies.iter().enumerate() {
            let lo = BARK_BAND_EDGES[b] as usize;
            let hi = BARK_BAND_EDGES[b + 1] as usize;
            if hi > lo {
                assert!(
                    *e > 0.0,
                    "band {b} of positive width must have non-zero energy"
                );
            } else {
                assert_eq!(*e, 0.0, "zero-width band {b} must contribute nothing");
            }
        }
    }

    #[test]
    fn bark_filterbank_rejects_wrong_magnitude_length() {
        let err = bark_filterbank(&[0.0f32; 5]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn interp_bark_gains_constant_input_yields_constant_output() {
        // A constant gain of 0.5 in every band linearly interpolates to
        // 0.5 at every STFT bin.
        let g = [0.5f32; N_BARK_BANDS];
        let interp = interp_bark_gains(&g).unwrap();
        assert_eq!(interp.len(), N_STFT_BINS);
        for (i, v) in interp.iter().enumerate() {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "constant gain must interp to constant: bin {i} = {v}"
            );
        }
    }

    #[test]
    fn interp_bark_gains_rejects_wrong_length() {
        let err = interp_bark_gains(&[0.0f32; 5]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn gru_forward_zero_input_zero_weights_leaves_state_unchanged() {
        // With W_ih = W_hh = 0 and b = 0, the reset / update gates are
        // sigmoid(0) = 0.5 and the new gate is tanh(0.5 * 0) = 0. Thus
        // h_new = (1 - 0.5) * 0 + 0.5 * h = 0.5 * h.
        let hidden = 3;
        let in_dim = 2;
        let mut state = vec![1.0f32; hidden];
        let w_ih = vec![0.0f32; 3 * hidden * in_dim];
        let w_hh = vec![0.0f32; 3 * hidden * hidden];
        let bias = vec![0.0f32; 3 * hidden];
        gru_forward(&[0.0, 0.0], &mut state, &w_ih, &w_hh, &bias).unwrap();
        for v in &state {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "expected h_new = 0.5 * h_prev, got {v}"
            );
        }
    }

    #[test]
    fn gru_forward_rejects_wrong_w_ih_shape() {
        let mut state = vec![0.0f32; 3];
        let w_ih = vec![0.0f32; 5]; // wrong: expected 3 * 3 * 2 = 18
        let w_hh = vec![0.0f32; 3 * 3 * 3];
        let bias = vec![0.0f32; 3 * 3];
        let err = gru_forward(&[0.0, 0.0], &mut state, &w_ih, &w_hh, &bias).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(msg) if msg.contains("w_ih")));
    }

    #[test]
    fn gru_forward_rejects_wrong_w_hh_shape() {
        let mut state = vec![0.0f32; 3];
        let w_ih = vec![0.0f32; 3 * 3 * 2];
        let w_hh = vec![0.0f32; 4]; // wrong
        let bias = vec![0.0f32; 3 * 3];
        let err = gru_forward(&[0.0, 0.0], &mut state, &w_ih, &w_hh, &bias).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(msg) if msg.contains("w_hh")));
    }

    #[test]
    fn gru_forward_rejects_wrong_bias_shape() {
        let mut state = vec![0.0f32; 3];
        let w_ih = vec![0.0f32; 3 * 3 * 2];
        let w_hh = vec![0.0f32; 3 * 3 * 3];
        let bias = vec![0.0f32; 2]; // wrong
        let err = gru_forward(&[0.0, 0.0], &mut state, &w_ih, &w_hh, &bias).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(msg) if msg.contains("bias")));
    }

    #[test]
    fn dense_forward_computes_y_eq_wx_plus_b_with_activation() {
        // 2 → 3 dense: W = [[1,0],[0,1],[1,1]], b = [0, 0, 0].
        // Identity activation: y = [x0, x1, x0 + x1].
        let w = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let b = vec![0.0, 0.0, 0.0];
        let y = dense_forward(&[3.0, 4.0], &w, &b, Activation::Identity).unwrap();
        assert_eq!(y, vec![3.0, 4.0, 7.0]);
    }

    #[test]
    fn dense_forward_rejects_wrong_weight_length() {
        let w = vec![0.0f32; 3]; // wrong: expected 3 * 2 = 6
        let b = vec![0.0f32; 3];
        let err = dense_forward(&[0.0, 0.0], &w, &b, Activation::Tanh).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(msg) if msg.contains("weight")));
    }

    #[test]
    fn pitch_analysis_returns_loud_partial() {
        let mut state = PitchState::default();
        let frame = vec![0.0f32; FRAME_SIZE];
        let err = pitch_analysis(&mut state, &frame).unwrap_err();
        let msg = match err {
            VokraError::UnsupportedOp(m) => m,
            other => panic!("expected UnsupportedOp, got {other:?}"),
        };
        assert!(
            msg.contains("VOKRA_RNNOISE_V02_REAL_GGUF"),
            "loud-partial must name the env-gate: {msg}"
        );
        assert!(
            msg.contains("parity_rnnoise_v02"),
            "loud-partial must name the parity script: {msg}"
        );
    }

    #[test]
    fn pack_features_layout_pin() {
        let bfcc = [1.0f32; N_BARK_BANDS];
        let pb = [2.0f32; N_PITCH_BANDS];
        let bd = [3.0f32; N_PITCH_BANDS];
        let pd = [4.0f32; N_PITCH_BANDS];
        let f = pack_features(&bfcc, &pb, &bd, &pd, 5.0, 6.0);
        assert_eq!(f.len(), N_FEATURES);
        for v in &f[..22] {
            assert_eq!(*v, 1.0);
        }
        for v in &f[22..28] {
            assert_eq!(*v, 2.0);
        }
        for v in &f[28..34] {
            assert_eq!(*v, 3.0);
        }
        for v in &f[34..40] {
            assert_eq!(*v, 4.0);
        }
        assert_eq!(f[40], 5.0);
        assert_eq!(f[41], 6.0);
    }

    #[test]
    fn zero_pitch_features_zeroes_all_pitch_slots() {
        let bfcc = [1.0f32; N_BARK_BANDS];
        let bd = [2.0f32; N_PITCH_BANDS];
        let f = zero_pitch_features(&bfcc, &bd);
        assert_eq!(f[22..28], [0.0f32; N_PITCH_BANDS]);
        assert_eq!(f[34..40], [0.0f32; N_PITCH_BANDS]);
        assert_eq!(f[40], 0.0);
        assert_eq!(f[41], 0.0);
    }

    #[test]
    fn bark_dct_shape_and_dc_pin() {
        // A constant input → DCT-II DC bin equals sum, higher bins are 0.
        let input = [1.0f32; N_BARK_BANDS];
        let out = bark_dct(&input);
        assert_eq!(out.len(), N_BARK_BANDS);
        assert!((out[0] - 22.0).abs() < 1e-5, "DC = sum, got {}", out[0]);
        // For a constant input the higher DCT-II coefficients are 0 (to
        // within FP noise on the cos-sum).
        for (k, v) in out.iter().enumerate().skip(1) {
            assert!(
                v.abs() < 1e-4,
                "DCT-II bin {k} of a constant input must be ~0, got {v}"
            );
        }
    }
}
