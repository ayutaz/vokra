//! `wpe` — Weighted Prediction Error blind dereverberation.
//!
//! # Primary source
//!
//! - Reference implementation: **`fgnt/nara_wpe`**, <https://github.com/fgnt/nara_wpe>
//!   — **MIT** ("Copyright (c) 2018 Communications Engineering Group, Paderborn
//!   University", `LICENSE`). Permissive, so this port is compatible with
//!   Vokra's Apache-2.0 posture (no GPL contamination — contrast `resample.rs`'s
//!   soxr / rubberband disclaimer, NFR-LC-03/04).
//! - Paper: Lukas Drude, Jahn Heymann, Christoph Boeddeker, Reinhold Haeb-Umbach,
//!   *"NARA-WPE: A Python package for weighted prediction error dereverberation
//!   in Numpy and Tensorflow for online and offline processing"*, ITG Fachtagung
//!   Sprachkommunikation (2018).
//! - Original method: Nakatani, Yoshioka, Kinoshita, Miyoshi, Juang,
//!   *"Speech Dereverberation Based on Variance-Normalized Delayed Linear
//!   Prediction"*, IEEE TASLP 18(7):1717-1731 (2010).
//!
//! Transcribed from the upstream `nara_wpe/wpe.py` **`wpe_v6`** variant (the
//! canonical offline iterative implementation). No upstream code is copied
//! verbatim — the numpy stride gymnastics are re-derived as explicit loops — but
//! the numerics follow it step for step, and the tap layout is pinned against
//! the upstream `build_y_tilde` doctest (see the
//! `build_y_tilde_matches_upstream_doctest` test).
//!
//! # Algorithm
//!
//! WPE models a reverberant STFT observation as a *delayed linear prediction*
//! problem, independently per frequency bin. Writing `Y[d, t]` for channel `d`
//! at frame `t`, the late reverberation at frame `t` is assumed to be a linear
//! function of the observation **at least `delay` frames in the past**:
//!
//! ```text
//! X[d, t] = Y[d, t] - Σ_r conj(G[r, d]) · Ỹ[r, t]
//! ```
//!
//! where `Ỹ` ("Y tilde") stacks `taps` delayed copies of every channel:
//!
//! ```text
//! Ỹ[k · D + d, t] = Y[d, t - delay - k]      (zero when t - delay - k < 0)
//! ```
//!
//! The `delay` is what makes this *blind*: the direct path and the early
//! reflections (which carry the speech) fall inside the excluded `delay`
//! window and are left untouched, while the predictable late tail is
//! subtracted.
//!
//! The prediction filter `G` is the solution of the **variance-normalized**
//! weighted normal equations `R · G = P`, with per-frame weights
//! `λ[t] = 1 / power[t]`:
//!
//! ```text
//! R[r, c] = Σ_t Ỹ[r, t] · λ[t] · conj(Ỹ[c, t])      ((taps·D) × (taps·D))
//! P[r, d] = Σ_t Ỹ[r, t] · λ[t] · conj(Y[d, t])      ((taps·D) × D)
//! ```
//!
//! (Derivation check: minimizing `Σ_t λ[t] · |Y[d,t] - Σ_r w_r Ỹ[r,t]|²` over
//! `w = conj(G[·, d])` gives `conj(P[c,d]) = Σ_r conj(G[r,d]) R[r,c]`;
//! conjugating and using `R` Hermitian yields exactly `P = R·G`. This is the
//! `G = _stable_solve(R, P)` / `X = Y - hermite(G) @ Y_tilde` pair upstream.)
//!
//! `power[t]` is the desired-signal variance estimate — the channel mean of
//! `|X[d, t]|²` — which is why the whole thing is iterated: each pass
//! re-estimates the power from the *current* dereverberated estimate and
//! re-solves. Upstream `wpe_v6`:
//!
//! ```text
//! X = copy(Y);  Ỹ = build_y_tilde(Y, taps, delay)
//! repeat `iterations` times:
//!     λ = get_power_inverse(X, psd_context)
//!     R, P = correlations(Ỹ, Y, λ)
//!     G = solve(R, P)
//!     X = Y - hermite(G) · Ỹ
//! ```
//!
//! # Upstream defaults (transcribed, not guessed)
//!
//! From the `wpe_v6` / `wpe_v7` signatures in `nara_wpe/wpe.py`:
//! `taps=10, delay=3, iterations=3, psd_context=0, statistics_mode='full'`.
//! They are re-exported here as [`DEFAULT_TAPS`], [`DEFAULT_DELAY`],
//! [`DEFAULT_ITERATIONS`], [`DEFAULT_PSD_CONTEXT`] and
//! [`StatisticsMode::Full`], and are what [`WpeAttrs::default`] installs.
//!
//! The variance floor is upstream's `_stable_positive_inverse`:
//! `eps = 1e-10 · max(power)`, then `1 / max(power, eps)`, and *all-ones* when
//! `eps == 0` (a digitally silent bin) — see [`POWER_INVERSE_EPS_FACTOR`].
//!
//! # Documented deviations from upstream
//!
//! 1. **`iterations == 0` is a loud error.** Upstream would silently return the
//!    input unchanged (the `for` loop simply does not run). Vokra rejects it
//!    with [`VokraError::InvalidArgument`] — a no-op dereverberator that
//!    reports success is exactly the silent-fallback failure mode FR-EX-08
//!    forbids. Same for `taps == 0`.
//! 2. **Rank-deficient solve.** Upstream is `try: np.linalg.solve(R, P);
//!    except LinAlgError: lstsq(...)`. `vokra-ops` carries no LAPACK, so
//!    `solve_psd_in_place` is a from-scratch complex Gaussian elimination
//!    with partial pivoting plus a *rank-revealing* branch: a column whose
//!    remaining modulus falls below the tolerance is a direction in which `Ỹ`
//!    carries no energy, so that component of `G` is set to zero. For a
//!    Hermitian PSD `R` this **is** the minimum-norm solution numpy's `lstsq`
//!    fallback returns: `R[r,r] = 0` forces `Ỹ[r,·] ≡ 0` (weights are
//!    non-negative), which forces `P[r,·] = 0`, so the system is consistent
//!    and the free component is genuinely arbitrary. Digitally silent bins
//!    therefore pass through untouched instead of producing `NaN`.
//! 3. **`psd_context`** accepts only the symmetric integer form. Upstream also
//!    accepts an asymmetric `(left, right)` tuple and `np.inf` (global mean
//!    over the whole utterance); neither is modelled here.
//! 4. **`ridge`** ([`WpeAttrs::ridge`]) is a Vokra-only optional relative
//!    diagonal load. It is **not** an upstream parameter and defaults to `0.0`,
//!    i.e. off, so the default path is numerically upstream's.
//!
//! # Runtime function, not an `OpKind` variant (ADR M4-20 §D-5)
//!
//! Like [`resample`](crate::resample), [`agc`](crate::agc) and [`hpf`](crate::hpf),
//! `wpe` is exposed as a first-class API function rather than an `OpKind` enum
//! variant / `dispatch.rs` arm, so a graph-side call falls into the existing
//! `UnsupportedOp` default (FR-EX-08). [`WpeAttrs`] is defined here (not in
//! `vokra-core`) for the same reason: it is not embedded in any `OpKind`.
//!
//! # Weight-free
//!
//! WPE is pure DSP — there are no learned parameters, no checkpoint, and
//! therefore no model licence, no `vokra.provenance.*` gate and no model-zoo
//! entry. Like [`resample`](crate::resample) it is available unconditionally.
//!
//! # Determinism / zero-dep
//!
//! Every accumulation runs in `f64` in a fixed index order, the Hermitian
//! `R` is built from its upper triangle and mirrored (so `R[c,r]` is *exactly*
//! `conj(R[r,c])`, not merely equal up to summation order), and there is no
//! RNG and no interior mutability. Two identical inputs therefore produce
//! bit-identical outputs (the `same_input_twice_is_bit_identical` test). Only
//! `std` is used — the complex arithmetic and the linear solver are both
//! defined in this module (NFR-DS-02).

use std::ops::{Add, Mul, Sub};

use vokra_core::ir::graph::{IstftAttrs, StftAttrs};
use vokra_core::{Result, VokraError};

use crate::{Spectrogram, istft, stft};

// ---------------------------------------------------------------------------
// Upstream defaults
// ---------------------------------------------------------------------------

/// Upstream `taps` default — the prediction filter order, i.e. how many
/// delayed frames per channel enter the regressor stack.
///
/// From `nara_wpe/wpe.py`: `def wpe_v6(Y, taps=10, delay=3, iterations=3, ...)`.
pub const DEFAULT_TAPS: usize = 10;

/// Upstream `delay` default — the prediction delay in frames. Frames closer
/// than this are excluded from the regressor, which is what protects the
/// direct path and the early reflections from being cancelled.
///
/// From `nara_wpe/wpe.py`: `def wpe_v6(Y, taps=10, delay=3, iterations=3, ...)`.
pub const DEFAULT_DELAY: usize = 3;

/// Upstream `iterations` default — how many times the power estimate is
/// refreshed from the current dereverberated estimate and the filter re-solved.
///
/// From `nara_wpe/wpe.py`: `def wpe_v6(Y, taps=10, delay=3, iterations=3, ...)`.
pub const DEFAULT_ITERATIONS: usize = 3;

/// Upstream `psd_context` default — half-width, in frames, of the centred
/// moving average applied to the power estimate. `0` means "no smoothing".
///
/// From `nara_wpe/wpe.py`: `def wpe_v6(..., psd_context=0, statistics_mode='full')`.
pub const DEFAULT_PSD_CONTEXT: usize = 0;

/// The relative variance floor from upstream `_stable_positive_inverse`:
///
/// ```text
/// eps = 1e-10 * np.max(power)
/// if eps == 0: inverse_power = np.ones_like(power)
/// else:        inverse_power = 1 / np.maximum(power, eps)
/// ```
pub const POWER_INVERSE_EPS_FACTOR: f64 = 1e-10;

/// Which frames contribute to the correlation statistics `R` and `P`.
///
/// Upstream `statistics_mode`; the default is `'full'`.
///
/// ```text
/// if statistics_mode == 'full':  s = Ellipsis
/// elif statistics_mode == 'valid': s = (Ellipsis, slice(delay + taps - 1, None))
/// ```
///
/// Note this selects which frames are *summed over* when estimating the filter;
/// the filter is always applied to every frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StatisticsMode {
    /// All frames contribute (upstream `'full'`, the default). The leading
    /// `delay + taps - 1` frames have a partly zero-padded regressor.
    #[default]
    Full,
    /// Only frames from `delay + taps - 1` onwards contribute (upstream
    /// `'valid'`), i.e. only those whose regressor stack is fully populated by
    /// real observations rather than the leading zero pad.
    Valid,
}

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

/// Attributes for [`wpe`] and its wrappers.
///
/// [`Default`] installs the upstream `wpe_v6` defaults verbatim
/// (`taps = 10`, `delay = 3`, `iterations = 3`, `psd_context = 0`,
/// `statistics_mode = 'full'`) plus `ridge = 0.0` (the Vokra-only diagonal
/// load, off).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WpeAttrs {
    /// Prediction filter order in frames, per channel. Must be `>= 1`
    /// ([`DEFAULT_TAPS`]).
    pub taps: usize,
    /// Prediction delay in frames. `0` is legal (upstream allows it) and makes
    /// the filter predict from the immediately preceding frame, which will also
    /// attack the direct path ([`DEFAULT_DELAY`]).
    pub delay: usize,
    /// Number of power-re-estimation passes. Must be `>= 1` — see the
    /// "Documented deviations" note in the [module docs](self)
    /// ([`DEFAULT_ITERATIONS`]).
    pub iterations: usize,
    /// Half-width, in frames, of the centred moving average smoothing applied
    /// to the power estimate. `0` disables smoothing ([`DEFAULT_PSD_CONTEXT`]).
    pub psd_context: usize,
    /// Which frames contribute to the correlation statistics
    /// ([`StatisticsMode`]).
    pub statistics_mode: StatisticsMode,
    /// **Not an upstream parameter.** Optional relative diagonal load added to
    /// `R` before the solve: `R[i,i] += ridge · trace(R) / n`. Scale-invariant
    /// (it is relative to the trace), and `0.0` — the default — leaves the
    /// numerics identical to upstream. Raise it only if a pathological input
    /// produces a visibly ill-conditioned solve. Must be finite and `>= 0`.
    pub ridge: f64,
}

impl Default for WpeAttrs {
    fn default() -> Self {
        Self {
            taps: DEFAULT_TAPS,
            delay: DEFAULT_DELAY,
            iterations: DEFAULT_ITERATIONS,
            psd_context: DEFAULT_PSD_CONTEXT,
            statistics_mode: StatisticsMode::Full,
            ridge: 0.0,
        }
    }
}

impl WpeAttrs {
    /// Upstream `wpe_v6` defaults (same as [`Default::default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Upstream defaults with an explicit `taps` / `delay`, for callers whose
    /// STFT hop differs from the 16 kHz / 512 / 128 setting the upstream
    /// defaults were chosen for.
    #[must_use]
    pub fn with_taps_delay(taps: usize, delay: usize) -> Self {
        Self {
            taps,
            delay,
            ..Self::default()
        }
    }

    /// Checks the attribute invariants.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when `taps == 0`, when `iterations == 0`
    /// (both are loud rather than silent no-ops — FR-EX-08), or when `ridge` is
    /// negative or non-finite.
    pub fn validate(&self) -> Result<()> {
        if self.taps == 0 {
            return Err(VokraError::InvalidArgument(
                "wpe: taps must be >= 1 (taps = 0 would leave the signal \
                 unchanged; a dereverberator that silently does nothing is \
                 rejected rather than reported as success — FR-EX-08)"
                    .to_owned(),
            ));
        }
        if self.iterations == 0 {
            return Err(VokraError::InvalidArgument(
                "wpe: iterations must be >= 1 (iterations = 0 would leave the \
                 signal unchanged; upstream nara_wpe silently returns the input \
                 in that case, Vokra rejects it — FR-EX-08)"
                    .to_owned(),
            ));
        }
        if !self.ridge.is_finite() || self.ridge < 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "wpe: ridge {} must be finite and >= 0",
                self.ridge
            )));
        }
        Ok(())
    }

    /// First frame index that contributes to the correlation statistics, given
    /// a clip of `n_frames` frames (clamped, so an over-long `delay + taps`
    /// yields an empty statistics window rather than an out-of-range index).
    fn stats_start(&self, n_frames: usize) -> usize {
        match self.statistics_mode {
            StatisticsMode::Full => 0,
            // Saturating so a pathological `delay`/`taps` clamps to "empty
            // statistics window" rather than wrapping. `taps >= 1` is
            // guaranteed by `validate`, which every entry point runs first.
            StatisticsMode::Valid => self
                .delay
                .saturating_add(self.taps)
                .saturating_sub(1)
                .min(n_frames),
        }
    }
}

// ---------------------------------------------------------------------------
// Double-precision complex scratch type
// ---------------------------------------------------------------------------

/// Double-precision complex number used for every WPE accumulation.
///
/// [`Complex32`](vokra_core::Complex32) is the `f32` host/IR type; WPE
/// forms Gram matrices and inverts them, so the intermediate arithmetic runs in
/// `f64` and is narrowed back to `f32` only when the result is written into the
/// output [`Spectrogram`]. Field order and operator shape mirror
/// `vokra_core::complex::Complex32` deliberately.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    const ZERO: Self = Self { re: 0.0, im: 0.0 };
    const ONE: Self = Self { re: 1.0, im: 0.0 };

    #[inline]
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// The complex conjugate `re - i·im`.
    #[inline]
    const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// The squared magnitude `re² + im²`.
    #[inline]
    fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Scales both components by the real scalar `s`.
    #[inline]
    fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    /// Complex division. Named `div_by` rather than implementing
    /// [`std::ops::Div`] purely so the module keeps a single obvious spot for
    /// the only operation that can divide by zero (the solver guards its
    /// pivots, so it never does).
    #[inline]
    fn div_by(self, rhs: Self) -> Self {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }

    /// Whether both components are exactly zero.
    #[inline]
    fn is_zero(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
}

impl Add for C64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for C64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for C64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

// ---------------------------------------------------------------------------
// Algorithm pieces
// ---------------------------------------------------------------------------

/// Builds the stacked delayed-observation matrix `Ỹ` for one frequency bin.
///
/// `y` is row-major `[n_channels][n_frames]`; `out` is filled row-major
/// `[taps · n_channels][n_frames]` with
///
/// ```text
/// out[k · n_channels + d][t] = y[d][t - delay - k]   (zero when t < delay + k)
/// ```
///
/// This layout — tap-major, channel-minor, lag increasing with the row block —
/// is the one upstream `build_y_tilde` produces; its docstring doctest is
/// replayed verbatim in the `build_y_tilde_matches_upstream_doctest` test.
///
/// The row ordering is in any case a *basis permutation* of the regressor
/// stack: permuting the rows of `Ỹ` permutes the rows of `R`/`P` and hence of
/// `G` consistently, leaving `X = Y - Gᴴ·Ỹ` invariant. Matching upstream
/// exactly still matters for anyone diffing `G` against `nara_wpe`.
fn build_y_tilde(
    y: &[C64],
    n_channels: usize,
    n_frames: usize,
    taps: usize,
    delay: usize,
    out: &mut Vec<C64>,
) {
    out.clear();
    out.resize(taps * n_channels * n_frames, C64::ZERO);
    for k in 0..taps {
        // Saturating: a caller-supplied `delay` near `usize::MAX` must land in
        // the "everything is zero-padded" branch, not wrap around into a
        // spurious in-range lag.
        let lag = delay.saturating_add(k);
        if lag >= n_frames {
            // Every entry of this tap block is zero-padded: nothing to copy.
            continue;
        }
        for d in 0..n_channels {
            let dst = (k * n_channels + d) * n_frames;
            let src = d * n_frames;
            for t in lag..n_frames {
                out[dst + t] = y[src + t - lag];
            }
        }
    }
}

/// Per-frame power: the channel mean of `|x[d][t]|²`.
///
/// Upstream `get_power_inverse`: `power = np.mean(abs_square(signal), axis=-2)`
/// where `signal` has shape `(D, T)` and `axis=-2` is the channel axis.
fn frame_power(x: &[C64], n_channels: usize, n_frames: usize, out: &mut Vec<f64>) {
    out.clear();
    out.resize(n_frames, 0.0);
    for d in 0..n_channels {
        let base = d * n_frames;
        for (o, v) in out.iter_mut().zip(&x[base..base + n_frames]) {
            *o += v.norm_sqr();
        }
    }
    let inv_d = 1.0 / n_channels as f64;
    for v in out.iter_mut() {
        *v *= inv_d;
    }
}

/// Centred moving average of `power` with half-width `context`, normalized by
/// the number of in-range taps at each position.
///
/// This is upstream `window_mean(power, (psd_context, psd_context))`: the
/// denominator is computed by running the same box filter over an array of
/// ones, so "at position 0 with left context 1, division is by 2 instead of 3".
fn window_mean_power(power: &[f64], context: usize, out: &mut Vec<f64>) {
    out.clear();
    out.resize(power.len(), 0.0);
    if power.is_empty() {
        return;
    }
    let n = power.len();
    for (t, o) in out.iter_mut().enumerate() {
        let lo = t.saturating_sub(context);
        let hi = t.saturating_add(context).min(n - 1);
        let mut acc = 0.0;
        for &v in &power[lo..=hi] {
            acc += v;
        }
        *o = acc / (hi - lo + 1) as f64;
    }
}

/// Upstream `_stable_positive_inverse`: floor the power at `1e-10 · max(power)`
/// then invert, and fall back to all-ones when that floor is exactly zero
/// (a digitally silent bin), which is what keeps silence from becoming `NaN`.
fn stable_positive_inverse(power: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.resize(power.len(), 1.0);
    let max = power.iter().copied().fold(0.0f64, f64::max);
    let eps = POWER_INVERSE_EPS_FACTOR * max;
    if eps == 0.0 {
        // `out` is already all ones.
        return;
    }
    for (o, &p) in out.iter_mut().zip(power) {
        *o = 1.0 / p.max(eps);
    }
}

/// Solves the Hermitian positive-semi-definite system `A · X = B` in place,
/// leaving `X` in `b`.
///
/// `a` is `n × n` row-major, `b` is `n × m` row-major. Complex Gaussian
/// elimination with partial pivoting, entirely self-contained (no LAPACK, no
/// external crate — NFR-DS-02).
///
/// Rank deficiency is handled rather than reported: when the remaining modulus
/// of column `k` falls at or below the tolerance, `Ỹ` carries no energy in that
/// direction. For a Hermitian PSD matrix `|a_ij|² <= a_ii · a_jj`, so a
/// vanishing diagonal forces the whole row and column to vanish, and (because
/// `P[r,·]` is built from the same rows of `Ỹ`) the right-hand side vanishes
/// with it. The system is therefore consistent and that component of the
/// solution is genuinely arbitrary; pinning it to zero reproduces the
/// minimum-norm answer numpy's `lstsq` fallback gives upstream.
///
/// `ridge` (when `> 0`) first adds `ridge · trace(A) / n` to the diagonal.
fn solve_psd_in_place(a: &mut [C64], b: &mut [C64], n: usize, m: usize, ridge: f64) {
    if n == 0 || m == 0 {
        return;
    }

    if ridge > 0.0 {
        let mut trace = 0.0f64;
        for i in 0..n {
            trace += a[i * n + i].re;
        }
        let load = ridge * trace / n as f64;
        if load > 0.0 {
            for i in 0..n {
                a[i * n + i].re += load;
            }
        }
    }

    // For a Hermitian PSD matrix the largest modulus sits on the (real,
    // non-negative) diagonal, so the diagonal is the right scale reference.
    let mut scale = 0.0f64;
    for i in 0..n {
        scale = scale.max(a[i * n + i].re.abs());
    }
    let tol = scale * f64::EPSILON * n as f64;

    for k in 0..n {
        // Partial pivoting on squared modulus (avoids n sqrt calls).
        let mut piv = k;
        let mut best = a[k * n + k].norm_sqr();
        for i in (k + 1)..n {
            let v = a[i * n + k].norm_sqr();
            if v > best {
                best = v;
                piv = i;
            }
        }

        if best.sqrt() <= tol {
            // Numerically empty direction: force this component of the
            // solution to zero (see the doc comment). Note the rows below are
            // already ~zero in this column, so there is nothing to eliminate.
            for j in k..n {
                a[k * n + j] = C64::ZERO;
            }
            a[k * n + k] = C64::ONE;
            for j in 0..m {
                b[k * m + j] = C64::ZERO;
            }
            continue;
        }

        if piv != k {
            for j in 0..n {
                a.swap(k * n + j, piv * n + j);
            }
            for j in 0..m {
                b.swap(k * m + j, piv * m + j);
            }
        }

        let pivot = a[k * n + k];
        for i in (k + 1)..n {
            let f = a[i * n + k].div_by(pivot);
            if f.is_zero() {
                continue;
            }
            a[i * n + k] = C64::ZERO;
            for j in (k + 1)..n {
                let t = f * a[k * n + j];
                a[i * n + j] = a[i * n + j] - t;
            }
            for j in 0..m {
                let t = f * b[k * m + j];
                b[i * m + j] = b[i * m + j] - t;
            }
        }
    }

    // Back substitution.
    for k in (0..n).rev() {
        let diag = a[k * n + k];
        for j in 0..m {
            let mut acc = b[k * m + j];
            for c in (k + 1)..n {
                acc = acc - a[k * n + c] * b[c * m + j];
            }
            b[k * m + j] = acc.div_by(diag);
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Runs offline iterative WPE dereverberation over a multi-channel
/// spectrogram.
///
/// `channels` is one [`Spectrogram`] per microphone — together the
/// `[n_channels][n_frames][n_freq]` complex observation. All channels must
/// agree on `frames` and `bins`. A single-element slice is a legal mono call
/// (see [`wpe_mono`]); WPE is well defined for `D = 1`, where it degenerates to
/// single-channel delayed linear prediction.
///
/// Returns one dereverberated [`Spectrogram`] per input channel, same shape.
///
/// Bins are processed independently, exactly as upstream.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] when [`WpeAttrs::validate`] fails, when
/// `channels` is empty, when the channels disagree on shape, when a
/// spectrogram's `re`/`im` length does not match `frames · bins`, when
/// `taps · n_channels` or a scratch length overflows `usize`, or when any input
/// sample is non-finite (FR-EX-08 — never a silent clamp).
pub fn wpe(channels: &[Spectrogram], attrs: &WpeAttrs) -> Result<Vec<Spectrogram>> {
    attrs.validate()?;

    if channels.is_empty() {
        return Err(VokraError::InvalidArgument(
            "wpe: needs at least one channel spectrogram".to_owned(),
        ));
    }

    let n_ch = channels.len();
    let n_frames = channels[0].frames;
    let n_bins = channels[0].bins;

    let expected = n_frames.checked_mul(n_bins).ok_or_else(|| {
        VokraError::InvalidArgument("wpe: frames * bins overflows usize".to_owned())
    })?;

    for (i, c) in channels.iter().enumerate() {
        if c.frames != n_frames || c.bins != n_bins {
            return Err(VokraError::InvalidArgument(format!(
                "wpe: channel {i} is {}x{} but channel 0 is {n_frames}x{n_bins} \
                 (all channels must share the STFT geometry)",
                c.frames, c.bins
            )));
        }
        if c.re.len() != expected || c.im.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "wpe: channel {i} has re/im lengths {}/{}, expected {expected} \
                 (= frames {n_frames} * bins {n_bins})",
                c.re.len(),
                c.im.len()
            )));
        }
        if c.re.iter().chain(c.im.iter()).any(|v| !v.is_finite()) {
            return Err(VokraError::InvalidArgument(format!(
                "wpe: channel {i} has a non-finite sample"
            )));
        }
    }

    if n_frames == 0 || n_bins == 0 {
        return Ok(channels.to_vec());
    }

    let p = attrs.taps.checked_mul(n_ch).ok_or_else(|| {
        VokraError::InvalidArgument("wpe: taps * n_channels overflows usize".to_owned())
    })?;
    // Guard the scratch allocations up front so a hostile attrs/shape pair
    // reports rather than aborting on a capacity overflow.
    p.checked_mul(n_frames)
        .and_then(|v| v.checked_add(p.checked_mul(p)?))
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "wpe: taps * n_channels * frames overflows usize".to_owned(),
            )
        })?;

    let t_start = attrs.stats_start(n_frames);

    // Output starts as a copy of the input; each bin overwrites its own column.
    let mut out: Vec<Spectrogram> = channels.to_vec();

    // Scratch reused across bins.
    let mut y: Vec<C64> = vec![C64::ZERO; n_ch * n_frames];
    let mut x: Vec<C64> = vec![C64::ZERO; n_ch * n_frames];
    let mut y_tilde: Vec<C64> = Vec::new();
    let mut y_tilde_ip: Vec<C64> = vec![C64::ZERO; p * n_frames];
    let mut power: Vec<f64> = Vec::new();
    let mut power_smoothed: Vec<f64> = Vec::new();
    let mut inv_power: Vec<f64> = Vec::new();
    let mut r_mat: Vec<C64> = vec![C64::ZERO; p * p];
    let mut p_mat: Vec<C64> = vec![C64::ZERO; p * n_ch];

    for bin in 0..n_bins {
        // Gather this bin's `[D][T]` observation.
        for (d, ch) in channels.iter().enumerate() {
            let base = d * n_frames;
            for t in 0..n_frames {
                let idx = t * n_bins + bin;
                y[base + t] = C64::new(ch.re[idx] as f64, ch.im[idx] as f64);
            }
        }

        build_y_tilde(&y, n_ch, n_frames, attrs.taps, attrs.delay, &mut y_tilde);
        x.copy_from_slice(&y);

        for _ in 0..attrs.iterations {
            // --- variance-normalized weights ---------------------------------
            frame_power(&x, n_ch, n_frames, &mut power);
            let pw = if attrs.psd_context > 0 {
                window_mean_power(&power, attrs.psd_context, &mut power_smoothed);
                &power_smoothed
            } else {
                &power
            };
            stable_positive_inverse(pw, &mut inv_power);

            // --- Ỹ · diag(λ) -------------------------------------------------
            for r in 0..p {
                let row = r * n_frames;
                for t in 0..n_frames {
                    y_tilde_ip[row + t] = y_tilde[row + t].scale(inv_power[t]);
                }
            }

            // --- R = Ỹ·diag(λ)·Ỹᴴ -------------------------------------------
            // Built from the upper triangle and mirrored, so R is *exactly*
            // Hermitian rather than Hermitian-up-to-summation-order. The
            // diagonal is formed as Σ|Ỹ|²λ, which is real by construction.
            for r in 0..p {
                let row_r = r * n_frames;
                let mut diag = 0.0f64;
                for t in t_start..n_frames {
                    diag += y_tilde[row_r + t].norm_sqr() * inv_power[t];
                }
                r_mat[r * p + r] = C64::new(diag, 0.0);

                for c in (r + 1)..p {
                    let row_c = c * n_frames;
                    let mut acc = C64::ZERO;
                    for t in t_start..n_frames {
                        acc = acc + y_tilde_ip[row_r + t] * y_tilde[row_c + t].conj();
                    }
                    r_mat[r * p + c] = acc;
                    r_mat[c * p + r] = acc.conj();
                }
            }

            // --- P = Ỹ·diag(λ)·Yᴴ -------------------------------------------
            for r in 0..p {
                let row_r = r * n_frames;
                for d in 0..n_ch {
                    let base = d * n_frames;
                    let mut acc = C64::ZERO;
                    for t in t_start..n_frames {
                        acc = acc + y_tilde_ip[row_r + t] * y[base + t].conj();
                    }
                    p_mat[r * n_ch + d] = acc;
                }
            }

            // --- G = R⁻¹ P, then X = Y - Gᴴ Ỹ -------------------------------
            solve_psd_in_place(&mut r_mat, &mut p_mat, p, n_ch, attrs.ridge);
            // `p_mat` now holds G, row-major [p][n_ch].

            for d in 0..n_ch {
                let base = d * n_frames;
                for t in 0..n_frames {
                    let mut acc = y[base + t];
                    for r in 0..p {
                        acc = acc - p_mat[r * n_ch + d].conj() * y_tilde[r * n_frames + t];
                    }
                    x[base + t] = acc;
                }
            }
        }

        // Scatter back.
        for (d, ch) in out.iter_mut().enumerate() {
            let base = d * n_frames;
            for t in 0..n_frames {
                let idx = t * n_bins + bin;
                let v = x[base + t];
                ch.re[idx] = v.re as f32;
                ch.im[idx] = v.im as f32;
            }
        }
    }

    Ok(out)
}

/// Single-channel convenience wrapper over [`wpe`].
///
/// # Errors
///
/// As [`wpe`].
pub fn wpe_mono(spec: &Spectrogram, attrs: &WpeAttrs) -> Result<Spectrogram> {
    let mut out = wpe(std::slice::from_ref(spec), attrs)?;
    // `wpe` returns exactly one spectrogram for a one-channel input.
    Ok(out.remove(0))
}

/// End-to-end mono convenience path: `stft` → [`wpe`] → `istft`.
///
/// Uses [`StftAttrs::new`] / [`IstftAttrs::new`] — the librosa/torch-like
/// defaults (periodic Hann, `win_length = n_fft`, `center = true` with reflect
/// padding, backward normalization, RFFT half-spectrum) — matched on both
/// sides, and pins the reconstruction length to `pcm.len()` so the output is
/// sample-aligned with the input.
///
/// Choose `n_fft` and `hop_length` together with [`WpeAttrs::delay`]: the
/// filter predicts from `delay · hop_length` samples in the past, so that
/// product should sit beyond the direct path and the early reflections but
/// inside the room's decay. Upstream's own examples use 16 kHz with
/// `n_fft = 512`, `hop_length = 128` and the [`DEFAULT_DELAY`] of 3.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] when `attrs` is invalid, when `pcm` contains
/// a non-finite sample, or as propagated from `stft` / `istft` (zero `n_fft`,
/// zero `hop_length`, ...).
pub fn wpe_dereverb_pcm(
    pcm: &[f32],
    attrs: &WpeAttrs,
    n_fft: usize,
    hop_length: usize,
) -> Result<Vec<f32>> {
    attrs.validate()?;
    if pcm.iter().any(|s| !s.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "wpe_dereverb_pcm: input has a non-finite sample".to_owned(),
        ));
    }

    let sa = StftAttrs::new(n_fft, hop_length);
    let spec = stft(pcm, &sa)?;
    let dereverbed = wpe_mono(&spec, attrs)?;

    let mut ia = IstftAttrs::new(n_fft, hop_length);
    ia.length = Some(pcm.len());
    istft(&dereverbed, &ia)
}

/// Multi-channel counterpart of [`wpe_dereverb_pcm`]: one time-domain buffer
/// per microphone in, one dereverberated buffer per microphone out.
///
/// This is the configuration WPE was designed for — the extra channels give the
/// prediction filter spatial as well as temporal support.
///
/// # Errors
///
/// As [`wpe_dereverb_pcm`], plus [`VokraError::InvalidArgument`] when `pcm` is
/// empty or the channels differ in length.
pub fn wpe_dereverb_pcm_multi(
    pcm: &[Vec<f32>],
    attrs: &WpeAttrs,
    n_fft: usize,
    hop_length: usize,
) -> Result<Vec<Vec<f32>>> {
    attrs.validate()?;
    if pcm.is_empty() {
        return Err(VokraError::InvalidArgument(
            "wpe_dereverb_pcm_multi: needs at least one channel".to_owned(),
        ));
    }
    let len = pcm[0].len();
    for (i, ch) in pcm.iter().enumerate() {
        if ch.len() != len {
            return Err(VokraError::InvalidArgument(format!(
                "wpe_dereverb_pcm_multi: channel {i} has {} samples, channel 0 \
                 has {len} (channels must be time-aligned and equal length)",
                ch.len()
            )));
        }
        if ch.iter().any(|s| !s.is_finite()) {
            return Err(VokraError::InvalidArgument(format!(
                "wpe_dereverb_pcm_multi: channel {i} has a non-finite sample"
            )));
        }
    }

    let sa = StftAttrs::new(n_fft, hop_length);
    let specs: Vec<Spectrogram> = pcm
        .iter()
        .map(|ch| stft(ch, &sa))
        .collect::<Result<Vec<_>>>()?;
    let dereverbed = wpe(&specs, attrs)?;

    let mut ia = IstftAttrs::new(n_fft, hop_length);
    ia.length = Some(len);
    dereverbed.iter().map(|s| istft(s, &ia)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- deterministic fixtures ------------------------------------------
    //
    // No committed fixture files: every signal below is generated in-test from
    // a fixed-seed 64-bit LCG (Knuth / MMIX multiplier), so the tests are
    // reproducible on every platform without touching the repo tree.

    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        /// Uniform in `[-1, 1)`.
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (self.0 >> 33) as f64 / (1u64 << 30) as f64; // [0, 2)
            (u - 1.0) as f32
        }
    }

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Lcg::new(seed);
        (0..n).map(|_| rng.next_f32()).collect()
    }

    /// A short exponentially-decaying synthetic room impulse response:
    /// unit direct path at n = 0, then a diffuse tail `0.5 · e^{-n/tau} · u`
    /// with `u` uniform in `[-1, 1)`.
    fn exp_rir(len: usize, tau: f64, seed: u64) -> Vec<f32> {
        let mut rng = Lcg::new(seed);
        let mut h = vec![0.0f32; len];
        h[0] = 1.0;
        for (n, v) in h.iter_mut().enumerate().skip(1) {
            let env = 0.5 * (-(n as f64) / tau).exp();
            *v = (env * rng.next_f32() as f64) as f32;
        }
        h
    }

    /// `x * h`, truncated to `x.len()` (a causal room response).
    fn convolve_truncated(x: &[f32], h: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; x.len()];
        for (n, yv) in y.iter_mut().enumerate() {
            let kmax = h.len().min(n + 1);
            let mut acc = 0.0f64;
            for (k, &hk) in h[..kmax].iter().enumerate() {
                acc += hk as f64 * x[n - k] as f64;
            }
            *yv = acc as f32;
        }
        y
    }

    /// A burst train: `cycles` repetitions of `on` noise samples followed by
    /// `off` silent samples. The silent stretches are where "late tail" energy
    /// is measured — with no source driving them, everything there is room.
    fn burst_train(cycles: usize, on: usize, off: usize, seed: u64) -> Vec<f32> {
        let mut rng = Lcg::new(seed);
        let mut out = Vec::with_capacity(cycles * (on + off));
        for _ in 0..cycles {
            for _ in 0..on {
                out.push(rng.next_f32());
            }
            out.resize(out.len() + off, 0.0);
        }
        out
    }

    fn energy(x: &[f32]) -> f64 {
        x.iter().map(|&v| v as f64 * v as f64).sum()
    }

    fn spec_from_pcm(pcm: &[f32], n_fft: usize, hop: usize) -> Spectrogram {
        stft(pcm, &StftAttrs::new(n_fft, hop)).unwrap()
    }

    /// Total complex energy of a spectrogram.
    fn spec_energy(s: &Spectrogram) -> f64 {
        s.re.iter()
            .zip(&s.im)
            .map(|(&r, &i)| r as f64 * r as f64 + i as f64 * i as f64)
            .sum()
    }

    // ---- upstream parity --------------------------------------------------

    #[test]
    fn build_y_tilde_matches_upstream_doctest() {
        // Replays the doctest in `nara_wpe/wpe.py::build_y_tilde` verbatim:
        //
        //   T, D = 20, 2
        //   Y = np.arange(start=1, stop=T * D + 1).reshape([T, D]).T
        //   taps, delay = 4, 2
        //   Y_tilde = build_y_tilde(Y, taps, delay)
        //
        // Y[d][t] == 2t + d + 1. The expected matrices below are the literal
        // printed output from the upstream docstring — this is the one place
        // the tap ordering / lag alignment is pinned against the primary
        // source rather than against our own formula.
        const T: usize = 20;
        const D: usize = 2;
        let mut y = vec![C64::ZERO; D * T];
        for d in 0..D {
            for t in 0..T {
                y[d * T + t] = C64::new((2 * t + d + 1) as f64, 0.0);
            }
        }

        // ---- delay = 2 ----
        #[rustfmt::skip]
        let expected_delay2: [[i32; T]; 8] = [
            [0, 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35],
            [0, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36],
            [0, 0, 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33],
            [0, 0, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
            [0, 0, 0, 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
            [0, 0, 0, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
            [0, 0, 0, 0, 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29],
            [0, 0, 0, 0, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
        ];
        let mut got = Vec::new();
        build_y_tilde(&y, D, T, 4, 2, &mut got);
        assert_eq!(got.len(), 8 * T, "Y_tilde shape must be (taps*D, T)");
        for (r, row) in expected_delay2.iter().enumerate() {
            for (t, &want) in row.iter().enumerate() {
                assert_eq!(
                    got[r * T + t],
                    C64::new(want as f64, 0.0),
                    "delay=2 mismatch at row {r}, frame {t}"
                );
            }
        }

        // ---- delay = 0 (second doctest block) ----
        #[rustfmt::skip]
        let expected_delay0: [[i32; T]; 8] = [
            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
            [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37],
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
            [0, 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35],
            [0, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36],
            [0, 0, 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33],
            [0, 0, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
        ];
        build_y_tilde(&y, D, T, 4, 0, &mut got);
        for (r, row) in expected_delay0.iter().enumerate() {
            for (t, &want) in row.iter().enumerate() {
                assert_eq!(
                    got[r * T + t],
                    C64::new(want as f64, 0.0),
                    "delay=0 mismatch at row {r}, frame {t}"
                );
            }
        }
    }

    #[test]
    fn stable_positive_inverse_matches_upstream_floor() {
        // eps = 1e-10 * max(power); values below the floor invert to 1/eps.
        let power = [1.0, 0.0, 1e-30, 0.25];
        let mut inv = Vec::new();
        stable_positive_inverse(&power, &mut inv);
        let eps = POWER_INVERSE_EPS_FACTOR * 1.0;
        assert_eq!(inv[0], 1.0);
        assert_eq!(
            inv[1],
            1.0 / eps,
            "zero power must clamp to the 1/eps floor"
        );
        assert_eq!(inv[2], 1.0 / eps, "sub-floor power must clamp too");
        assert_eq!(inv[3], 4.0);

        // All-zero power => eps == 0 => all ones (the silence guard: without
        // it every silent bin would be 1/0 = inf and then NaN).
        let mut inv0 = Vec::new();
        stable_positive_inverse(&[0.0, 0.0, 0.0], &mut inv0);
        assert_eq!(inv0, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn window_mean_power_normalizes_at_the_edges() {
        // Upstream `window_mean` divides by the number of in-range taps, so at
        // position 0 with context 1 the divisor is 2, not 3. Hand-derived
        // oracle for [1,2,3,4,5], context 1:
        //   t=0: (1+2)/2   = 1.5
        //   t=1: (1+2+3)/3 = 2
        //   t=2: (2+3+4)/3 = 3
        //   t=3: (3+4+5)/3 = 4
        //   t=4: (4+5)/2   = 4.5
        let mut out = Vec::new();
        window_mean_power(&[1.0, 2.0, 3.0, 4.0, 5.0], 1, &mut out);
        assert_eq!(out, vec![1.5, 2.0, 3.0, 4.0, 4.5]);

        // Context 0 is the identity.
        let mut id = Vec::new();
        window_mean_power(&[1.0, 2.0, 3.0], 0, &mut id);
        assert_eq!(id, vec![1.0, 2.0, 3.0]);
    }

    // ---- solver -----------------------------------------------------------

    #[test]
    fn solver_recovers_a_known_solution_on_a_full_rank_hermitian_system() {
        // A = M·Mᴴ + 3I is Hermitian positive definite by construction.
        const N: usize = 4;
        const M: usize = 2;
        let mut rng = Lcg::new(0xA11CE);
        let mut m = vec![C64::ZERO; N * N];
        for v in m.iter_mut() {
            *v = C64::new(rng.next_f32() as f64, rng.next_f32() as f64);
        }
        let mut a = vec![C64::ZERO; N * N];
        for i in 0..N {
            for j in 0..N {
                let mut acc = C64::ZERO;
                for k in 0..N {
                    acc = acc + m[i * N + k] * m[j * N + k].conj();
                }
                a[i * N + j] = acc;
            }
            a[i * N + i] = a[i * N + i] + C64::new(3.0, 0.0);
        }

        // Known X, then B = A·X.
        let mut x_true = vec![C64::ZERO; N * M];
        for v in x_true.iter_mut() {
            *v = C64::new(rng.next_f32() as f64, rng.next_f32() as f64);
        }
        let mut b = vec![C64::ZERO; N * M];
        for i in 0..N {
            for j in 0..M {
                let mut acc = C64::ZERO;
                for k in 0..N {
                    acc = acc + a[i * N + k] * x_true[k * M + j];
                }
                b[i * M + j] = acc;
            }
        }

        let mut x = b;
        solve_psd_in_place(&mut a, &mut x, N, M, 0.0);

        for (got, want) in x.iter().zip(&x_true) {
            assert!(
                (got.re - want.re).abs() < 1e-10 && (got.im - want.im).abs() < 1e-10,
                "solver did not recover the known solution: got {got:?}, want {want:?}"
            );
        }
    }

    #[test]
    fn solver_handles_a_rank_deficient_system() {
        // A = v·vᴴ is rank 1 (and singular for N > 1). With B = A·x0 the
        // system is consistent, so a valid solve must reproduce B — that is
        // the oracle, not the (non-unique) solution vector itself.
        const N: usize = 3;
        let v = [
            C64::new(1.0, 0.5),
            C64::new(-2.0, 0.25),
            C64::new(0.75, -1.5),
        ];
        let mut a = vec![C64::ZERO; N * N];
        for i in 0..N {
            for j in 0..N {
                a[i * N + j] = v[i] * v[j].conj();
            }
        }
        let x0 = [C64::new(0.3, -0.7), C64::new(1.1, 0.2), C64::new(-0.4, 0.9)];
        let mut b = vec![C64::ZERO; N];
        for i in 0..N {
            let mut acc = C64::ZERO;
            for k in 0..N {
                acc = acc + a[i * N + k] * x0[k];
            }
            b[i] = acc;
        }

        let mut a_work = a.clone();
        let mut x = b.clone();
        solve_psd_in_place(&mut a_work, &mut x, N, 1, 0.0);

        assert!(
            x.iter().all(|v| v.re.is_finite() && v.im.is_finite()),
            "a singular system must not produce NaN/inf: {x:?}"
        );
        for i in 0..N {
            let mut acc = C64::ZERO;
            for k in 0..N {
                acc = acc + a[i * N + k] * x[k];
            }
            let scale = b[i].norm_sqr().sqrt().max(1.0);
            assert!(
                (acc - b[i]).norm_sqr().sqrt() < 1e-9 * scale,
                "rank-deficient solve must still satisfy A·x = b at row {i}: \
                 got {acc:?}, want {:?}",
                b[i]
            );
        }
    }

    #[test]
    fn solver_on_an_all_zero_system_yields_zero() {
        // The silent-bin path: A = 0, B = 0 => x = 0 (never NaN from 0/0).
        const N: usize = 3;
        let mut a = vec![C64::ZERO; N * N];
        let mut b = vec![C64::ZERO; N];
        solve_psd_in_place(&mut a, &mut b, N, 1, 0.0);
        for v in &b {
            assert_eq!(*v, C64::ZERO, "all-zero system must solve to exactly zero");
        }
    }

    // ---- exact structural oracles ----------------------------------------

    #[test]
    fn digital_silence_passes_through_exactly() {
        // All-zero input: power == 0 everywhere, so `stable_positive_inverse`
        // takes its eps == 0 branch (all-ones weights), R and P are zero, the
        // solver returns G = 0, and X = Y - 0 = 0. Exact, and it proves the
        // silence path never manufactures a NaN.
        let spec = Spectrogram {
            frames: 24,
            bins: 9,
            re: vec![0.0; 24 * 9],
            im: vec![0.0; 24 * 9],
        };
        let out = wpe_mono(&spec, &WpeAttrs::with_taps_delay(4, 2)).unwrap();
        assert!(out.re.iter().all(|&v| v == 0.0), "silence must stay silent");
        assert!(out.im.iter().all(|&v| v == 0.0), "silence must stay silent");
    }

    #[test]
    fn delay_past_the_clip_is_a_bit_exact_passthrough() {
        // When `delay >= n_frames` every lag is out of range, so Ỹ is
        // identically zero => R = 0, P = 0 => G = 0 => X == Y exactly. This
        // pins the lag indexing: an off-by-one in `build_y_tilde` would leak a
        // non-zero regressor here and perturb the output.
        let pcm = noise(4096, 7);
        let spec = spec_from_pcm(&pcm, 128, 64);
        let attrs = WpeAttrs {
            delay: spec.frames + 1,
            taps: 3,
            iterations: 2,
            ..WpeAttrs::default()
        };
        let out = wpe_mono(&spec, &attrs).unwrap();
        assert_eq!(out.re, spec.re, "out-of-range delay must be a passthrough");
        assert_eq!(out.im, spec.im, "out-of-range delay must be a passthrough");
    }

    #[test]
    fn weighted_residual_never_increases() {
        // G is by definition the minimizer of
        //     J(G) = Σ_t λ[t] · ‖Y[:,t] - Gᴴ·Ỹ[:,t]‖²
        // and G = 0 is a feasible point with J(0) = Σ_t λ[t]·‖Y[:,t]‖².
        // Hence J(G) <= J(0) *exactly* — a derived inequality, not a tuned
        // bound. (With `ridge > 0` the minimized objective is J(G)+ridge‖G‖²,
        // so J(G) <= J(0) still holds; here ridge = 0.)
        //
        // `iterations = 1` and `psd_context = 0` make λ come from X = Y, i.e.
        // straight from the input, so the test can rebuild the exact weights.
        let pcm = convolve_truncated(&noise(8192, 11), &exp_rir(512, 200.0, 12));
        let spec = spec_from_pcm(&pcm, 128, 64);
        let attrs = WpeAttrs {
            taps: 5,
            delay: 2,
            iterations: 1,
            psd_context: 0,
            statistics_mode: StatisticsMode::Full,
            ridge: 0.0,
        };
        let out = wpe_mono(&spec, &attrs).unwrap();

        let (frames, bins) = (spec.frames, spec.bins);
        let mut j_y = 0.0f64;
        let mut j_x = 0.0f64;
        let mut y = vec![C64::ZERO; frames];
        let mut power = Vec::new();
        let mut inv = Vec::new();
        for bin in 0..bins {
            for (t, yt) in y.iter_mut().enumerate() {
                let idx = t * bins + bin;
                *yt = C64::new(spec.re[idx] as f64, spec.im[idx] as f64);
            }
            frame_power(&y, 1, frames, &mut power);
            stable_positive_inverse(&power, &mut inv);
            for t in 0..frames {
                let idx = t * bins + bin;
                let xv = C64::new(out.re[idx] as f64, out.im[idx] as f64);
                j_y += inv[t] * y[t].norm_sqr();
                j_x += inv[t] * xv.norm_sqr();
            }
        }

        // Slack covers only the f32 quantization of the returned spectrogram
        // (the arithmetic itself is f64).
        assert!(
            j_x <= j_y * (1.0 + 1e-4),
            "the weighted residual must not increase: J(G) = {j_x}, J(0) = {j_y}"
        );
        // ...and it must actually have done something on a reverberant input,
        // so the inequality above is not passing trivially.
        assert!(
            j_x < j_y * 0.99,
            "WPE should measurably reduce the weighted residual on reverberant \
             input: J(G) = {j_x}, J(0) = {j_y}"
        );
    }

    #[test]
    fn same_input_twice_is_bit_identical() {
        let pcm = convolve_truncated(&noise(4096, 21), &exp_rir(384, 150.0, 22));
        let spec = spec_from_pcm(&pcm, 128, 64);
        let attrs = WpeAttrs {
            taps: 4,
            delay: 2,
            iterations: 3,
            psd_context: 2,
            statistics_mode: StatisticsMode::Valid,
            ridge: 0.0,
        };
        let a = wpe_mono(&spec, &attrs).unwrap();
        let b = wpe_mono(&spec, &attrs).unwrap();
        assert_eq!(a.re, b.re, "WPE must be bit-deterministic");
        assert_eq!(a.im, b.im, "WPE must be bit-deterministic");

        // ...and so must the time-domain wrapper.
        let p1 = wpe_dereverb_pcm(&pcm, &attrs, 128, 64).unwrap();
        let p2 = wpe_dereverb_pcm(&pcm, &attrs, 128, 64).unwrap();
        assert_eq!(p1, p2, "wpe_dereverb_pcm must be bit-deterministic");
    }

    // ---- functional behaviour --------------------------------------------

    #[test]
    fn late_reverberation_tail_energy_decreases() {
        // The real functional test. A burst train (2048 samples of noise, then
        // 2048 samples of silence) is convolved with an exponentially decaying
        // RIR. Inside a silent stretch the source contributes nothing, so all
        // remaining energy there *is* late reverberation — and by construction
        // it is a linear function of the observation further in the past,
        // which is exactly what WPE's delayed linear predictor removes.
        //
        // Geometry: hop = 128, delay = 2 => the predictor draws on samples
        // 256..(2+6-1)·128 = 256..896 in the past. The RIR decay constant is
        // 500 samples and it runs for RIR_LEN = 1536, which is shorter than
        // the 2048-sample gap — so the measured window
        // (gap_start+320 .. gap_start+1536) is entirely reverberation, still
        // carries energy across its whole span, and sits beyond the `delay`
        // guard that protects the direct path.
        const ON: usize = 2048;
        const OFF: usize = 2048;
        const CYCLES: usize = 4;
        const N_FFT: usize = 1024;
        const HOP: usize = 256;
        const RIR_LEN: usize = 1536;

        // ---- why these parameters, and not smaller ones ------------------
        //
        // A WPE filter predicts frame `t` from frames `t-delay` through
        // `t-delay-taps+1`, so its reach into the past is
        //
        //     reach_samples = (delay + taps) * hop
        //
        // Suppressing reverberation at offset `n` into a silent gap requires
        // the filter to still see the burst that caused it, i.e. `reach >= n`.
        // This fixture measures the tail out to `RIR_LEN = 1536` samples, so
        // any configuration with `reach < 1536` is physically unable to
        // cancel most of the measured window — it is not a weak result, it is
        // an unreachable one.
        //
        // Measured on this exact fixture (tail-energy ratio, iterations = 3):
        //
        //     n_fft  hop  taps  delay   reach   ratio
        //       256  128     6      2    1024  0.9997   <- unreachable
        //       256  128    10      2    1536  0.9967
        //       256  128    30      2    4096  0.5068
        //       512  256    10      2    3072  0.7611
        //      1024  256    10      2    3072  0.4657
        //      1024  256    20      2    5632  0.3045   <- chosen
        //      1024  256    30      2    8192  0.2630
        //
        // The chosen row has reach 5632 >= 1536 with margin, and the longer
        // analysis window also lets each frame observe more of the tail.
        const TAPS: usize = 20;
        const DELAY: usize = 2;
        // Compile-time: every operand is a const, so this guards the fixture
        // at build time rather than at run time. If someone shrinks TAPS/HOP
        // or grows RIR_LEN below the reach requirement, the crate stops
        // building with this message instead of the test silently becoming
        // an unreachable-and-therefore-meaningless assertion.
        const _: () = assert!(
            (DELAY + TAPS) * HOP >= RIR_LEN,
            "fixture/params mismatch: the WPE reach (delay + taps) * hop \
             cannot cover the RIR_LEN-sample tail this test measures"
        );

        let src = burst_train(CYCLES, ON, OFF, 31);
        let rir = exp_rir(RIR_LEN, 500.0, 32);
        let reverberant = convolve_truncated(&src, &rir);

        let attrs = WpeAttrs {
            taps: TAPS,
            delay: DELAY,
            iterations: 3,
            psd_context: 0,
            statistics_mode: StatisticsMode::Full,
            ridge: 0.0,
        };
        let dereverbed = wpe_dereverb_pcm(&reverberant, &attrs, N_FFT, HOP).unwrap();
        assert_eq!(dereverbed.len(), reverberant.len());

        // Sum the late portion of every silent gap.
        let mut tail_in = 0.0f64;
        let mut tail_out = 0.0f64;
        for c in 0..CYCLES {
            let gap_start = c * (ON + OFF) + ON;
            let lo = gap_start + 320;
            let hi = (gap_start + RIR_LEN).min(reverberant.len());
            if lo >= hi {
                continue;
            }
            tail_in += energy(&reverberant[lo..hi]);
            tail_out += energy(&dereverbed[lo..hi]);
        }

        assert!(
            tail_in > 0.0,
            "fixture is broken: the unprocessed gaps carry no reverb energy"
        );
        let ratio = tail_out / tail_in;
        // Conservative threshold: WPE should do far better than a 15 % energy
        // reduction on this fixture, but the bound is deliberately loose so it
        // pins the *direction* of the effect (late tail strictly suppressed)
        // without becoming a brittle regression target. Tighten it only
        // alongside a recorded measurement.
        assert!(
            ratio < 0.85,
            "WPE must suppress the late reverberation tail: \
             tail energy ratio {ratio} (in {tail_in}, out {tail_out})"
        );
    }

    #[test]
    fn anechoic_input_passes_through_largely_unchanged() {
        // "Identity-ish" bound, derived rather than tuned.
        //
        // Fitting p = taps · D complex regressors to T frames of an
        // incompressible (white) observation removes, in expectation, the
        // fraction p/T of the energy — the standard result that projecting an
        // isotropic T-dimensional vector onto a p-dimensional subspace
        // captures p/T of its energy. There is nothing else for WPE to find in
        // an anechoic signal, so p/T is the floor it should sit near.
        //
        // Geometry matters for the "white" premise: with n_fft = 256, hop =
        // 128 and delay = 2, frame t and frame t-2 share *no* samples
        // (delay·hop = 256 = n_fft), so the analysis window itself introduces
        // no correlation at the lags the predictor uses.
        const N: usize = 16384;
        const N_FFT: usize = 256;
        const HOP: usize = 128;

        let anechoic = noise(N, 41);
        let rir = exp_rir(1536, 500.0, 42);
        let reverberant = convolve_truncated(&anechoic, &rir);

        let attrs = WpeAttrs {
            taps: 4,
            delay: 2,
            iterations: 2,
            psd_context: 0,
            statistics_mode: StatisticsMode::Full,
            ridge: 0.0,
        };

        let spec_a = spec_from_pcm(&anechoic, N_FFT, HOP);
        let spec_r = spec_from_pcm(&reverberant, N_FFT, HOP);
        let out_a = wpe_mono(&spec_a, &attrs).unwrap();
        let out_r = wpe_mono(&spec_r, &attrs).unwrap();

        let removed_a = 1.0 - spec_energy(&out_a) / spec_energy(&spec_a);
        let removed_r = 1.0 - spec_energy(&out_r) / spec_energy(&spec_r);

        // p/T for this configuration.
        let nominal = attrs.taps as f64 / spec_a.frames as f64;
        // Allowance over the nominal p/T: the projection is weighted (so not
        // exactly orthogonal), it is re-solved across `iterations` passes, and
        // the Hann window's spectral leakage leaves a little inter-frame
        // correlation even at non-overlapping lags. 10x covers all three.
        let bound = 10.0 * nominal;
        assert!(
            removed_a < bound,
            "anechoic input should sit near the p/T projection floor: removed \
             {removed_a}, nominal p/T {nominal}, bound {bound}"
        );

        // The comparative statement is the load-bearing one: WPE has to find
        // substantially more to remove in a reverberant signal than in an
        // anechoic one, or it is not dereverberating at all. The factor is
        // deliberately conservative — for this RIR roughly half the energy
        // sits beyond the delay·hop = 256-sample guard, so the true separation
        // is far wider than 2x, but pinning the *direction* is what matters
        // here (`late_reverberation_tail_energy_decreases` is the quantitative
        // functional test).
        assert!(
            removed_r > 2.0 * removed_a,
            "reverberant input must be affected far more than anechoic: \
             removed_reverberant {removed_r} vs removed_anechoic {removed_a}"
        );
    }

    #[test]
    fn multichannel_matches_mono_shape_and_beats_the_residual_bound() {
        // Two microphones observing the same source through different rooms:
        // the multi-channel solve has D = 2, so p = taps·2 regressors.
        let src = noise(6144, 51);
        let a = convolve_truncated(&src, &exp_rir(512, 220.0, 52));
        let b = convolve_truncated(&src, &exp_rir(512, 220.0, 53));
        let attrs = WpeAttrs {
            taps: 3,
            delay: 2,
            iterations: 2,
            ..WpeAttrs::default()
        };
        let out = wpe_dereverb_pcm_multi(&[a.clone(), b.clone()], &attrs, 128, 64).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), a.len());
        assert_eq!(out[1].len(), b.len());
        assert!(
            out.iter().all(|c| c.iter().all(|v| v.is_finite())),
            "multi-channel output must be finite"
        );

        // Both channels must have been *substantially* modified. A bare
        // `!=` would pass on the STFT/iSTFT roundtrip error alone (~1e-6
        // relative), so the bound is set three orders of magnitude above it:
        // anything below 1 % relative change means the filter did nothing and
        // WPE silently degenerated into a resynthesis.
        for (i, (dereverbed, original)) in out.iter().zip([&a, &b]).enumerate() {
            let diff: f64 = dereverbed
                .iter()
                .zip(original.iter())
                .map(|(&o, &r)| {
                    let d = o as f64 - r as f64;
                    d * d
                })
                .sum();
            let rel = (diff / energy(original)).sqrt();
            assert!(
                rel > 0.01,
                "channel {i} must be dereverberated, not resynthesized: \
                 relative change {rel}"
            );
        }
    }

    #[test]
    fn statistics_mode_valid_changes_the_estimate() {
        // `Valid` drops the leading `delay + taps - 1` frames from the
        // statistics, so on a short clip the two modes must disagree.
        let pcm = convolve_truncated(&noise(3072, 61), &exp_rir(256, 120.0, 62));
        let spec = spec_from_pcm(&pcm, 128, 64);
        let base = WpeAttrs {
            taps: 4,
            delay: 3,
            iterations: 2,
            ..WpeAttrs::default()
        };
        let full = wpe_mono(
            &spec,
            &WpeAttrs {
                statistics_mode: StatisticsMode::Full,
                ..base
            },
        )
        .unwrap();
        let valid = wpe_mono(
            &spec,
            &WpeAttrs {
                statistics_mode: StatisticsMode::Valid,
                ..base
            },
        )
        .unwrap();
        assert_ne!(
            full.re, valid.re,
            "statistics_mode must actually select different frames"
        );

        // And when the statistics window is empty (delay+taps-1 >= frames),
        // Valid degenerates to a bit-exact passthrough.
        let short = WpeAttrs {
            taps: spec.frames,
            delay: spec.frames,
            iterations: 1,
            statistics_mode: StatisticsMode::Valid,
            ..WpeAttrs::default()
        };
        let out = wpe_mono(&spec, &short).unwrap();
        assert_eq!(out.re, spec.re);
        assert_eq!(out.im, spec.im);
    }

    #[test]
    fn psd_context_smoothing_changes_the_estimate() {
        let pcm = convolve_truncated(&noise(4096, 71), &exp_rir(384, 180.0, 72));
        let spec = spec_from_pcm(&pcm, 128, 64);
        let base = WpeAttrs {
            taps: 4,
            delay: 2,
            iterations: 2,
            ..WpeAttrs::default()
        };
        let sharp = wpe_mono(&spec, &base).unwrap();
        let smooth = wpe_mono(
            &spec,
            &WpeAttrs {
                psd_context: 3,
                ..base
            },
        )
        .unwrap();
        assert_ne!(
            sharp.re, smooth.re,
            "psd_context must actually smooth the variance estimate"
        );
    }

    // ---- loud failures ----------------------------------------------------

    #[test]
    fn zero_taps_is_a_loud_error() {
        let attrs = WpeAttrs {
            taps: 0,
            ..WpeAttrs::default()
        };
        let err = attrs.validate().unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "taps = 0 must be InvalidArgument, got {err:?}"
        );
        let spec = Spectrogram {
            frames: 4,
            bins: 3,
            re: vec![0.0; 12],
            im: vec![0.0; 12],
        };
        assert!(wpe_mono(&spec, &attrs).is_err(), "taps = 0 must not no-op");
        assert!(wpe_dereverb_pcm(&[0.0; 512], &attrs, 128, 64).is_err());
    }

    #[test]
    fn zero_iterations_is_a_loud_error() {
        let attrs = WpeAttrs {
            iterations: 0,
            ..WpeAttrs::default()
        };
        let err = attrs.validate().unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "iterations = 0 must be InvalidArgument, got {err:?}"
        );
        let spec = Spectrogram {
            frames: 4,
            bins: 3,
            re: vec![0.0; 12],
            im: vec![0.0; 12],
        };
        assert!(
            wpe_mono(&spec, &attrs).is_err(),
            "iterations = 0 must not silently pass the signal through"
        );
    }

    #[test]
    fn bad_ridge_is_a_loud_error() {
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            let attrs = WpeAttrs {
                ridge: bad,
                ..WpeAttrs::default()
            };
            assert!(
                matches!(attrs.validate(), Err(VokraError::InvalidArgument(_))),
                "ridge {bad} must be rejected"
            );
        }
        // A legal positive ridge still runs.
        let pcm = noise(2048, 81);
        let spec = spec_from_pcm(&pcm, 128, 64);
        let ok = WpeAttrs {
            taps: 3,
            delay: 2,
            iterations: 1,
            ridge: 1e-6,
            ..WpeAttrs::default()
        };
        let out = wpe_mono(&spec, &ok).unwrap();
        assert!(out.re.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn shape_and_finiteness_violations_are_loud() {
        let attrs = WpeAttrs::with_taps_delay(3, 2);

        // Empty channel list.
        assert!(matches!(
            wpe(&[], &attrs),
            Err(VokraError::InvalidArgument(_))
        ));

        // Mismatched geometry between channels.
        let a = Spectrogram {
            frames: 8,
            bins: 5,
            re: vec![0.0; 40],
            im: vec![0.0; 40],
        };
        let b = Spectrogram {
            frames: 8,
            bins: 4,
            re: vec![0.0; 32],
            im: vec![0.0; 32],
        };
        assert!(matches!(
            wpe(&[a.clone(), b], &attrs),
            Err(VokraError::InvalidArgument(_))
        ));

        // Declared shape does not match the buffer length.
        let ragged = Spectrogram {
            frames: 8,
            bins: 5,
            re: vec![0.0; 39],
            im: vec![0.0; 40],
        };
        assert!(matches!(
            wpe(&[ragged], &attrs),
            Err(VokraError::InvalidArgument(_))
        ));

        // Non-finite sample.
        let mut nan = a.clone();
        nan.im[7] = f32::NAN;
        assert!(matches!(
            wpe(&[nan], &attrs),
            Err(VokraError::InvalidArgument(_))
        ));

        // Non-finite PCM through the time-domain wrappers.
        assert!(wpe_dereverb_pcm(&[0.0, f32::INFINITY, 0.0], &attrs, 128, 64).is_err());
        assert!(wpe_dereverb_pcm_multi(&[], &attrs, 128, 64).is_err());
        assert!(
            wpe_dereverb_pcm_multi(&[vec![0.0; 512], vec![0.0; 256]], &attrs, 128, 64).is_err(),
            "ragged multi-channel input must be rejected"
        );
    }

    #[test]
    fn empty_spectrogram_is_handled() {
        let empty = Spectrogram {
            frames: 0,
            bins: 5,
            re: Vec::new(),
            im: Vec::new(),
        };
        let out = wpe_mono(&empty, &WpeAttrs::with_taps_delay(3, 2)).unwrap();
        assert_eq!(out.frames, 0);
        assert!(out.re.is_empty());
    }

    #[test]
    fn defaults_match_upstream() {
        let d = WpeAttrs::default();
        assert_eq!(d.taps, 10, "upstream nara_wpe wpe_v6 taps default");
        assert_eq!(d.delay, 3, "upstream nara_wpe wpe_v6 delay default");
        assert_eq!(
            d.iterations, 3,
            "upstream nara_wpe wpe_v6 iterations default"
        );
        assert_eq!(d.psd_context, 0, "upstream nara_wpe psd_context default");
        assert_eq!(d.statistics_mode, StatisticsMode::Full);
        assert_eq!(
            d.ridge, 0.0,
            "the Vokra-only diagonal load is off by default"
        );
        assert_eq!(d, WpeAttrs::new());
        d.validate().unwrap();
    }
}
