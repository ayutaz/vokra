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
//! - [`pitch_analysis`] — **real** autocorrelation-based pitch tracker
//!   (48 kHz native, [`MIN_LAG_SAMPLES`]..=[`MAX_LAG_SAMPLES`] search,
//!   parabolic sub-sample refinement, per-band correlation histogram
//!   for feature packing). Streaming: `PitchState` rolls the PCM
//!   lookback forward across calls. See [`pitch_analysis`] for the
//!   honest divergence from upstream Xiph (48 kHz vs 12 kHz analysis;
//!   no `remove_doubling` octave correction — the state carries
//!   `prev_period` / `prev_gain` so a downstream binder can layer one on).
//!
//! # FR-EX-08 loud-fail contract
//!
//! Every shape / dim mismatch is a hard error
//! ([`VokraError::InvalidArgument`]) naming the offending dimension.
//! [`pitch_analysis`] rejects a zero-length frame loudly rather than
//! silently short-circuiting to a fabricated `0.0`.

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

/// Analysable pitch-period band at the 48 kHz native rate — 60 Hz (long
/// vocal fry / bass) corresponds to a 800-sample period, but the RNNoise
/// downsampled buffer is only 720 samples (upstream downsamples to 12 kHz
/// = 200-sample buffer). Vokra runs autocorrelation directly at 48 kHz
/// over the 720-sample buffer, so the honest analysable band is
/// `[SAMPLE_RATE/MAX_LAG_SAMPLES, SAMPLE_RATE/MIN_LAG_SAMPLES]` Hz.
pub const MIN_LAG_SAMPLES: usize = 96; // 48 kHz / 96 = 500 Hz (upper F0)
/// Maximum autocorrelation lag in samples. `MAX_LAG_SAMPLES < PITCH_BUF_SIZE`
/// is required so at least `PITCH_BUF_SIZE - MAX_LAG_SAMPLES` samples
/// overlap at the maximum lag (otherwise the correlation window is empty).
/// 48 kHz / 700 = ~68 Hz (below adult male F0 floor).
pub const MAX_LAG_SAMPLES: usize = 700;

/// Runs one frame of pitch analysis (**real, autocorrelation-based**).
///
/// # Returns
///
/// `(pitch_period_samples, pitch_gain, [N_PITCH_BANDS] per-band pitch
/// correlation)` where:
///
/// - `pitch_period_samples` is the estimated pitch period at the 48 kHz
///   native rate (a period of `T` samples → fundamental frequency
///   `SAMPLE_RATE / T` Hz). `0.0` when no confident pitch is detected
///   (unvoiced frame — `pitch_gain < 0.0`).
/// - `pitch_gain` is the normalized autocorrelation peak at the winning
///   lag (`Σ x[n] * x[n-τ] / sqrt(Σ x[n]² · Σ x[n-τ]²)`), clamped into
///   `[0, 1]` when voiced.
/// - The per-band array carries the normalized correlation at 6 lags
///   spread across the pitch range — an approximation of the upstream
///   `NB_BANDS_PITCH` per-band pitch strength for feature packing.
///
/// # Streaming contract
///
/// The `state` PCM lookback is rolled forward by `frame.len()` samples
/// on every call — the caller does not need to manage the ring. The
/// autocorrelation runs against the concatenation of prior samples in
/// the lookback and the incoming `frame`, giving inter-frame continuity
/// so a slowly-changing pitch tracks across the 720-sample analysis
/// window.
///
/// # Design notes (honest divergence from upstream Xiph RNNoise)
///
/// Upstream `src/pitch.c` first downsamples the input by 4× (48 kHz →
/// 12 kHz) via two half-band filters and then runs the correlation on
/// the downsampled buffer — cheaper (fewer taps) and marginally more
/// robust because the low-pass rejects any partials above 6 kHz that
/// would alias into the pitch band. This runtime keeps the analysis at
/// 48 kHz to preserve `zero-dep` (no FIR filter tables) and adds a
/// simple parabolic interpolation between the argmax and its two
/// neighbours to recover sub-sample lag precision. The full upstream
/// `remove_doubling` post-processor (octave-error correction against the
/// prior frame's period) is not implemented: `state.prev_period` /
/// `prev_gain` are updated so a downstream binder can layer that on if
/// needed, but the raw argmax is returned so a caller sees an honest
/// per-frame estimate.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] on an empty `frame` (a zero-length
///   push has no honest semantic).
pub fn pitch_analysis(
    state: &mut PitchState,
    frame: &[f32],
) -> Result<(f32, f32, [f32; N_PITCH_BANDS])> {
    if frame.is_empty() {
        return Err(VokraError::InvalidArgument(
            "rnnoise::pitch_analysis: frame is empty; a zero-length push has no \
             honest semantic (FR-EX-08)"
                .into(),
        ));
    }

    // 1. Roll the lookback forward. The RNNoise `PITCH_BUF_SIZE` is the
    //    upper bound on how much history the autocorrelation walks — we
    //    keep exactly that much (oldest samples drop first once the
    //    buffer saturates).
    state.buffer.extend_from_slice(frame);
    if state.buffer.len() > PITCH_BUF_SIZE {
        let overflow = state.buffer.len() - PITCH_BUF_SIZE;
        state.buffer.drain(..overflow);
    }
    let buf = &state.buffer;

    // 2. If we do not yet have enough samples to run one full lag, return
    //    a "no confident pitch" reading. The RNNoise upstream also
    //    produces an unvoiced label until its buffer fills — the tracker
    //    is inherently a streaming algorithm.
    if buf.len() <= MIN_LAG_SAMPLES {
        state.prev_period = 0.0;
        state.prev_gain = 0.0;
        return Ok((0.0, 0.0, [0.0f32; N_PITCH_BANDS]));
    }

    // 3. Total-energy denominator across the analysis window. Reused for
    //    every lag in the search — precomputing once is O(N) instead of
    //    O(N · L).
    let n_analyze = buf.len();
    let mut e0 = 0.0f64;
    for &s in buf.iter() {
        e0 += f64::from(s) * f64::from(s);
    }
    // A silent buffer has no pitch; guard against the div-by-zero
    // that would otherwise appear in the normalization.
    if e0 < 1e-20 {
        state.prev_period = 0.0;
        state.prev_gain = 0.0;
        return Ok((0.0, 0.0, [0.0f32; N_PITCH_BANDS]));
    }

    // 4. Autocorrelation-based lag search. For each candidate lag τ,
    //    compute the normalized cross-correlation
    //      c[τ] = Σ_{n=τ}^{N-1} x[n] * x[n-τ]
    //             / sqrt(Σ_{n=τ}^{N-1} x[n]² · Σ_{n=0}^{N-1-τ} x[n]²)
    //    the denominator is the geometric mean of the two overlapping-
    //    window energies (the standard "normalized autocorrelation" that
    //    gives a value in [-1, 1] — a periodic signal at period τ hits
    //    +1). Range: MIN_LAG_SAMPLES..=lag_max.
    let lag_max = MAX_LAG_SAMPLES.min(n_analyze - 1);
    if lag_max < MIN_LAG_SAMPLES {
        state.prev_period = 0.0;
        state.prev_gain = 0.0;
        return Ok((0.0, 0.0, [0.0f32; N_PITCH_BANDS]));
    }

    let mut best_lag: usize = MIN_LAG_SAMPLES;
    let mut best_norm: f64 = -1.0;
    // Store the normalized correlation across the full search range —
    // needed for the per-band correlation pack and parabolic refinement.
    let mut corr_by_lag = vec![0.0f64; lag_max + 1];

    for tau in MIN_LAG_SAMPLES..=lag_max {
        let mut num = 0.0f64;
        let mut e_head = 0.0f64; // energy of x[τ..N]
        let mut e_tail = 0.0f64; // energy of x[0..N-τ]
        // Length of the overlapping window at this lag.
        let m = n_analyze - tau;
        for n in 0..m {
            let a = f64::from(buf[n + tau]);
            let b = f64::from(buf[n]);
            num += a * b;
            e_head += a * a;
            e_tail += b * b;
        }
        let denom = (e_head * e_tail).sqrt();
        // A near-zero denominator can happen for a lag near buf.len()
        // (only one or two overlapping samples); skip rather than divide.
        let norm = if denom > 1e-20 { num / denom } else { 0.0 };
        corr_by_lag[tau] = norm;
        if norm > best_norm {
            best_norm = norm;
            best_lag = tau;
        }
    }

    // 5. Parabolic refinement: fit a parabola through the argmax and its
    //    two immediate neighbours, take the vertex. Recovers sub-sample
    //    lag precision without more FLOPs than three subtracts and a
    //    divide. Skips the refinement if the argmax sits at the boundary
    //    of the search range (where there is no left / right neighbour).
    let refined_lag = if best_lag > MIN_LAG_SAMPLES && best_lag < lag_max {
        let cm = corr_by_lag[best_lag - 1];
        let c0 = corr_by_lag[best_lag];
        let cp = corr_by_lag[best_lag + 1];
        let denom = 2.0 * c0 - cm - cp;
        if denom.abs() > 1e-12 {
            best_lag as f64 + 0.5 * (cm - cp) / denom
        } else {
            best_lag as f64
        }
    } else {
        best_lag as f64
    };

    // 6. Voicing decision: the normalized correlation lies in [-1, 1]
    //    for real signals; a weak / noise-dominated frame produces peaks
    //    well below the typical voiced range (~0.3+). Upstream Xiph
    //    RNNoise uses a state-dependent threshold; we use a plain floor
    //    to keep the loud-partial → loud-real transition surface small.
    //    An unvoiced frame reports `pitch_period = 0.0` and `gain = 0.0`.
    const VOICED_GAIN_MIN: f64 = 0.30;
    let voiced = best_norm >= VOICED_GAIN_MIN;

    // 7. Per-band pitch correlation. Split the lag search range into 6
    //    contiguous bands and report the peak normalized correlation in
    //    each. This is the honest per-band feature the upstream RNNoise
    //    packs into `pack_features(pitch_bands, ...)` — a lag histogram
    //    that the classifier learns to interpret.
    //
    // Iteration by band index `b` (rather than `.iter_mut().enumerate()`)
    // keeps the band-relative `lo` / `hi` lag arithmetic aligned with
    // the mathematical definition; the enumerate rewrite would obscure
    // both.
    let mut pitch_bands = [0.0f32; N_PITCH_BANDS];
    let band_width = (lag_max - MIN_LAG_SAMPLES + 1) as f64 / N_PITCH_BANDS as f64;
    #[allow(clippy::needless_range_loop)]
    for b in 0..N_PITCH_BANDS {
        let lo = MIN_LAG_SAMPLES + (b as f64 * band_width).floor() as usize;
        let hi = MIN_LAG_SAMPLES + ((b + 1) as f64 * band_width).floor() as usize;
        let hi = hi.min(lag_max + 1);
        let lo = lo.min(hi);
        let mut peak = 0.0f64;
        // Explicit lag index `tau` — matches the corr_by_lag[τ] reading
        // in the upstream `celt_pitch_xcorr` reference so a reader can
        // cross-reference `src/pitch.c` line by line.
        for corr in corr_by_lag.iter().take(hi).skip(lo) {
            if *corr > peak {
                peak = *corr;
            }
        }
        pitch_bands[b] = peak.clamp(0.0, 1.0) as f32;
    }

    let (period, gain) = if voiced {
        let g = best_norm.clamp(0.0, 1.0) as f32;
        (refined_lag as f32, g)
    } else {
        (0.0f32, 0.0f32)
    };

    // 8. Persist for downstream `remove_doubling` layers if needed.
    state.prev_period = period;
    state.prev_gain = gain;

    Ok((period, gain, pitch_bands))
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
    fn pitch_analysis_rejects_empty_frame() {
        let mut state = PitchState::default();
        let err = pitch_analysis(&mut state, &[]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn pitch_analysis_silent_input_reports_unvoiced() {
        // A silent buffer must produce no confident pitch (voicing gate).
        // The invariant is `pitch_period == 0.0 && gain == 0.0`, never
        // a fabricated non-zero lag on pure silence.
        let mut state = PitchState::default();
        // Fill the buffer with enough zeros so the lag search actually
        // runs (otherwise we short-circuit before reaching the voicing
        // check).
        let frame = vec![0.0f32; PITCH_BUF_SIZE];
        let (period, gain, bands) = pitch_analysis(&mut state, &frame).unwrap();
        assert_eq!(period, 0.0, "silence must be unvoiced (period = 0)");
        assert_eq!(gain, 0.0, "silence must have zero gain");
        for &b in &bands {
            assert_eq!(b, 0.0, "silent input must produce zero band correlations");
        }
    }

    #[test]
    fn pitch_analysis_pure_tone_recovers_period_within_grid() {
        // A pure sine at 200 Hz sampled at 48 kHz has period 240 samples.
        // The autocorrelation-based tracker must land within a small
        // window of that period once the buffer fills (parabolic
        // interpolation gives sub-sample precision, so we tolerate 1
        // sample of drift for numerical noise).
        let mut state = PitchState::default();
        let f0 = 200.0f32;
        let sr = SAMPLE_RATE as f32;
        let expected_period = sr / f0; // 240
        // Two full buffer-fills worth of samples so the analysis window
        // is full and the parabolic refinement has meaningful neighbours.
        let n = PITCH_BUF_SIZE * 2;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI_F32 * f0 * (i as f32) / sr).sin())
            .collect();

        // Push the whole signal in one shot — the rolling buffer clamps
        // itself to PITCH_BUF_SIZE.
        let (period, gain, _bands) = pitch_analysis(&mut state, &signal).unwrap();
        let delta = (period - expected_period).abs();
        assert!(
            delta < 1.5,
            "expected period ≈ {expected_period}, got {period} (Δ = {delta}) — \
             autocorrelation-based pitch tracker must land within 1.5 samples"
        );
        assert!(
            gain > 0.9,
            "pure-tone gain must be near +1 (got {gain}) — a strong periodic \
             signal must trip the voicing gate"
        );
    }

    #[test]
    fn pitch_analysis_streams_state_across_calls() {
        // Two 100 ms slices of the same 200 Hz sine, pushed one at a
        // time, must land at the same period as a single-shot push. This
        // pins the streaming-state invariant so a future refactor cannot
        // silently regress into a per-call-only tracker.
        let f0 = 200.0f32;
        let sr = SAMPLE_RATE as f32;
        let slice_len = 4800; // 100 ms at 48 kHz
        let slice_a: Vec<f32> = (0..slice_len)
            .map(|i| (2.0 * PI_F32 * f0 * (i as f32) / sr).sin())
            .collect();
        let slice_b: Vec<f32> = (0..slice_len)
            .map(|i| (2.0 * PI_F32 * f0 * ((i + slice_len) as f32) / sr).sin())
            .collect();

        let mut s1 = PitchState::default();
        pitch_analysis(&mut s1, &slice_a).unwrap();
        let (p1, g1, _) = pitch_analysis(&mut s1, &slice_b).unwrap();
        assert!(p1 > 0.0, "streamed pitch must be voiced after two slices");
        assert!(g1 > 0.5, "streamed pitch must have positive gain, got {g1}");
        // State must be non-empty and clamped.
        assert!(!s1.buffer.is_empty());
        assert!(s1.buffer.len() <= PITCH_BUF_SIZE);
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
