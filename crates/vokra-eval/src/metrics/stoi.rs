//! Short-Time Objective Intelligibility (STOI) — the reference-based
//! **intelligibility** metric for enhancement / separation front-ends.
//!
//! STOI predicts how much of a clean signal's speech intelligibility survives a
//! processing chain. It is the standard companion to SI-SNR
//! ([`super::si_snr`]): SI-SNR says "how much of the waveform is left", STOI
//! says "how much of the *message* is left", and enhancement papers report
//! both because a denoiser can improve one while destroying the other.
//!
//! # Primary sources
//!
//! - Taal, Hendriks, Heusdens & Jensen, *"A short-time objective intelligibility
//!   measure for time-frequency weighted noisy speech"*, ICASSP 2010,
//!   <https://ieeexplore.ieee.org/document/5495701>;
//! - the same authors, *"An Algorithm for Intelligibility Prediction of
//!   Time–Frequency Weighted Noisy Speech"*, IEEE TASLP 19(7), 2011,
//!   <https://ieeexplore.ieee.org/document/5713237>;
//! - the parameter values below are the ones the authors' own reference release
//!   ships, cross-checked against the widely used `pystoi` port
//!   (<https://github.com/mpariente/pystoi>). No upstream source is vendored —
//!   this is an independent Rust transcription of the published algorithm, so
//!   no third-party licence attaches (NFR-DS-02 stays intact: the crate's
//!   dependency set is unchanged).
//!
//! # Algorithm
//!
//! 1. **Resample** both signals to 10 kHz (via [`vokra_ops::resample()`]).
//! 2. **Silence removal**: frame the *clean* signal at 256 / 128 with the Hann
//!    window, keep only frames whose energy is within `DYN_RANGE = 40 dB` of the
//!    loudest frame, and overlap-add the kept frames of *both* signals back to
//!    back. Silent regions carry no intelligibility and would otherwise
//!    dominate the average.
//! 3. **TF decomposition**: 256-sample Hann frames, hop 128, zero-padded to a
//!    512-point real FFT (via [`vokra_ops::fft::RealFftPlan`]).
//! 4. **One-third-octave bands**: 15 bands whose centres are `150 · 2^(k/3)` Hz;
//!    each band value is `sqrt(Σ |X|²)` over the band's FFT bins.
//! 5. **Segments**: every run of `N = 30` consecutive frames (≈ 384 ms).
//! 6. **Normalise + clip**: scale the degraded band envelope to the clean
//!    segment's energy, then clip it at `x·(1 + 10^(−β/20))` with `β = −15 dB`,
//!    bounding how much a single loud distortion can hurt.
//! 7. **Score**: the sample correlation coefficient of each clean/clipped pair,
//!    averaged over every (band, segment). Higher is better; ≈ 1.0 means the
//!    envelope structure survived intact.
//!
//! # Deliberate divergences from the reference release, and why
//!
//! - **Resampler.** The reference uses Octave/MATLAB `resample`; Vokra uses its
//!   own Kaiser-windowed-sinc [`vokra_ops::resample()`] (GPL-free by
//!   construction — soxr/rubberband are excluded, see CLAUDE.md). Scores on
//!   non-10 kHz input therefore agree closely but are **not** bit-exact with
//!   `pystoi`. Feed 10 kHz audio to take the resampler out of the comparison.
//! - **"Too short" is an error, not a sentinel.** When fewer than 30 frames
//!   survive silence removal the reference warns and returns `1e-5`. A sentinel
//!   that looks like a score is exactly the failure mode this repo refuses
//!   (FR-EX-08): a corpus runner would average `1e-5` into a mean and report a
//!   quiet lie. Vokra returns [`VokraError::InvalidArgument`] instead.
//! - **Zero-padding side.** The reference appends the 256 zeros after the
//!   windowed frame; the framing here is explicit and does the same. (Had this
//!   gone through [`vokra_ops::stft()`] the window would have been *centred* in
//!   the 512-point buffer — a pure phase rotation that STOI's magnitude-only
//!   band step cannot see, but the framing would then have been driven by
//!   `n_fft = 512` rather than the 256-sample analysis frame and would have
//!   dropped edge frames. That is why the frame loop is written out here while
//!   the FFT, the window and the resampler are all reused rather than
//!   reimplemented.)
//! - **Band-limited input.** Resampling *up* from below 10 kHz leaves the top
//!   one-third-octave bands empty, which depresses the score. That is inherent
//!   to running STOI on band-limited audio, not a bug; it is not rejected.
//!
//! # Extended STOI (ESTOI) is not implemented
//!
//! Jensen & Taal's ESTOI (2016) replaces the per-band correlation with a
//! whole-spectrum one and needs a different normalisation; it is a separate
//! metric, not a flag, and no wave currently reports it.

use super::{AudioRefMetric, Direction, Metric};
use vokra_core::ir::graph::{Window, WindowSymmetry};
use vokra_core::{Result, VokraError};
use vokra_ops::fft::RealFftPlan;
use vokra_ops::resample::{DEFAULT_QUALITY, resample};
use vokra_ops::window::window;

/// Analysis rate: STOI is defined at 10 kHz.
const FS: u32 = 10_000;
/// Analysis frame length in samples (25.6 ms at [`FS`]).
const N_FRAME: usize = 256;
/// Frame hop — 50 % overlap.
const HOP: usize = N_FRAME / 2;
/// Zero-padded FFT size (finer bin spacing for the band edges).
const NFFT: usize = 512;
/// Number of one-third-octave bands.
const NUM_BANDS: usize = 15;
/// Centre frequency of the lowest band, in Hz.
const MIN_FREQ: f64 = 150.0;
/// Frames per intelligibility segment (≈ 384 ms).
const SEG_FRAMES: usize = 30;
/// Lower SDR bound used by the clipping step, in dB.
const BETA_DB: f64 = -15.0;
/// Speech dynamic range for silence removal, in dB.
const DYN_RANGE_DB: f64 = 40.0;

/// Short-Time Objective Intelligibility (see the module docs).
///
/// The score is in `[-1, 1]` by construction (it is a mean of correlation
/// coefficients); intelligible speech lands near `1.0`. Higher is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stoi;

impl Stoi {
    /// Builds the metric (its analysis parameters are the fixed values the
    /// published algorithm specifies, so it carries no configuration).
    pub fn new() -> Self {
        Self
    }

    /// The analysis rate the metric resamples to, in Hz.
    pub fn analysis_rate(&self) -> u32 {
        FS
    }

    /// Scores `degraded` against `clean` (mono PCM, both at `sample_rate`).
    ///
    /// # Errors
    ///
    /// See [`stoi()`].
    pub fn score(&self, degraded: &[f32], clean: &[f32], sample_rate: u32) -> Result<f64> {
        stoi(degraded, clean, sample_rate)
    }
}

impl Metric for Stoi {
    fn name(&self) -> &str {
        "stoi"
    }
    fn direction(&self) -> Direction {
        Direction::HigherIsBetter
    }
}

impl AudioRefMetric for Stoi {
    fn eval_audio(&self, hyp: &[f32], reference: &[f32], sample_rate: u32) -> Result<f64> {
        stoi(hyp, reference, sample_rate)
    }
}

/// Computes STOI for `degraded` against `clean`, both mono PCM at
/// `sample_rate`.
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`] if the two buffers differ in length,
/// either is empty, `sample_rate` is `0`, a sample is non-finite, or fewer than
/// `SEG_FRAMES` (30) analysis frames survive silence removal (see the module docs
/// on why that last case is an error rather than the reference's `1e-5`
/// sentinel). Propagates any [`vokra_ops::resample()`] error.
pub fn stoi(degraded: &[f32], clean: &[f32], sample_rate: u32) -> Result<f64> {
    if degraded.len() != clean.len() {
        return Err(VokraError::InvalidArgument(format!(
            "stoi: length mismatch — degraded has {} samples, clean has {}. \
             Both must cover the same span; this metric does not truncate to \
             the shorter buffer.",
            degraded.len(),
            clean.len()
        )));
    }
    if clean.is_empty() {
        return Err(VokraError::InvalidArgument("stoi: empty input".to_owned()));
    }
    if sample_rate == 0 {
        return Err(VokraError::InvalidArgument(
            "stoi: sample_rate must be non-zero".to_owned(),
        ));
    }
    if let Some(bad) = clean.iter().chain(degraded).find(|v| !v.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "stoi: input contains a non-finite sample ({bad}) — refusing to \
             return a NaN score"
        )));
    }

    // (1) Resample to the 10 kHz analysis rate.
    let (clean, degraded) = if sample_rate == FS {
        (clean.to_vec(), degraded.to_vec())
    } else {
        (
            resample(clean, sample_rate, FS, DEFAULT_QUALITY)?,
            resample(degraded, sample_rate, FS, DEFAULT_QUALITY)?,
        )
    };
    let n = clean.len().min(degraded.len());
    let (clean, degraded) = (&clean[..n], &degraded[..n]);

    // The reference analysis window is MATLAB `hanning(256)` / numpy
    // `np.hanning(258)[1:-1]`: the SYMMETRIC Hann of length 258 with its two
    // zero endpoints trimmed. `vokra_ops::window` documents its symmetric form
    // as the `numpy.hanning` parity reference, so slicing it reproduces the
    // reference window exactly rather than approximating it with a periodic
    // Hann of length 256 (which is a different window).
    let full = window(Window::Hann, N_FRAME + 2, WindowSymmetry::Symmetric);
    let w: &[f32] = &full[1..=N_FRAME];

    // (2) Silence removal, driven by the clean signal's frame energies.
    let (clean, degraded) = remove_silent_frames(clean, degraded, w);

    // (3)+(4) Per-frame one-third-octave band envelopes, `[band][frame]`.
    let bands = third_octave_bands();
    let plan = RealFftPlan::new(NFFT);
    let (frames, clean_tob) = band_envelopes(&clean, w, &bands, &plan);
    let (deg_frames, deg_tob) = band_envelopes(&degraded, w, &bands, &plan);
    debug_assert_eq!(frames, deg_frames, "both signals are framed identically");

    if frames < SEG_FRAMES {
        return Err(VokraError::InvalidArgument(format!(
            "stoi: only {frames} analysis frame(s) survive silence removal, but \
             an intelligibility segment needs {SEG_FRAMES} (≈ {:.0} ms of \
             non-silent speech at {FS} Hz). The reference implementation warns \
             and returns a 1e-5 sentinel here; Vokra reports an error instead so \
             a corpus average cannot silently absorb it.",
            ((SEG_FRAMES - 1) * HOP + N_FRAME) as f64 * 1000.0 / f64::from(FS)
        )));
    }

    // (5)+(6)+(7) Segment, normalise, clip, correlate, average.
    let clip = 10f64.powf(-BETA_DB / 20.0);
    let segments = frames - SEG_FRAMES + 1;
    let mut acc = 0.0f64;
    let mut x = [0.0f64; SEG_FRAMES];
    let mut y = [0.0f64; SEG_FRAMES];
    for seg in 0..segments {
        for b in 0..bands.len() {
            let row = b * frames + seg;
            x.copy_from_slice(&clean_tob[row..row + SEG_FRAMES]);
            y.copy_from_slice(&deg_tob[row..row + SEG_FRAMES]);
            acc += clipped_correlation(&mut x, &mut y, clip);
        }
    }
    Ok(acc / (segments * bands.len()) as f64)
}

/// One (band, segment) intermediate intelligibility term: energy-normalise the
/// degraded envelope onto the clean one, clip it at `x·(1 + clip)`, and return
/// the sample correlation coefficient of the pair.
///
/// `x` and `y` are consumed as scratch (both are mean-removed in place).
fn clipped_correlation(x: &mut [f64; SEG_FRAMES], y: &mut [f64; SEG_FRAMES], clip: f64) -> f64 {
    // The reference guards every division with the f64 machine epsilon rather
    // than branching, so an all-zero band contributes exactly 0 instead of NaN.
    // Keeping that shape keeps the scores comparable.
    let eps = f64::EPSILON;
    let nx = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let ny = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    let alpha = nx / (ny + eps);
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = (*yi * alpha).min(xi * (1.0 + clip));
    }

    let mean_x = x.iter().sum::<f64>() / SEG_FRAMES as f64;
    let mean_y = y.iter().sum::<f64>() / SEG_FRAMES as f64;
    let mut num = 0.0f64;
    let mut dx = 0.0f64;
    let mut dy = 0.0f64;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let (a, b) = (xi - mean_x, yi - mean_y);
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    num / ((dx.sqrt() + eps) * (dy.sqrt() + eps))
}

/// Drops frames whose *clean* energy is more than [`DYN_RANGE_DB`] below the
/// loudest frame, then overlap-adds the kept (windowed) frames of both signals
/// back to back.
///
/// Frame starts are `0, HOP, 2·HOP, …` while `start + N_FRAME < len` — strictly
/// less, matching the reference's `range(0, len(x) - framelen, hop)` (the frame
/// that would end exactly at `len` is not analysed).
fn remove_silent_frames(clean: &[f32], degraded: &[f32], w: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let starts: Vec<usize> = (0..)
        .map(|i| i * HOP)
        .take_while(|s| s + N_FRAME < clean.len())
        .collect();
    if starts.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // 20·log10(‖w·x‖₂ + eps) per frame. The reference's extra 1/sqrt(N) factor
    // is a constant offset and cancels against the max, so it is omitted.
    let energies: Vec<f64> = starts
        .iter()
        .map(|&s| {
            let e: f64 = clean[s..s + N_FRAME]
                .iter()
                .zip(w)
                .map(|(&v, &wk)| {
                    let p = f64::from(v) * f64::from(wk);
                    p * p
                })
                .sum();
            20.0 * (e.sqrt() + f64::EPSILON).log10()
        })
        .collect();
    let loudest = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // `filter_map` (not `filter` + `map`) so the closure destructures the item
    // BY VALUE — under edition 2024's match ergonomics a `&`-pattern inside a
    // `filter` closure (whose argument is `&Item`) is a hard error.
    let kept: Vec<usize> = starts
        .iter()
        .zip(&energies)
        .filter_map(|(&s, &e)| (loudest - DYN_RANGE_DB - e < 0.0).then_some(s))
        .collect();
    if kept.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let out_len = (kept.len() - 1) * HOP + N_FRAME;
    let mut clean_out = vec![0.0f32; out_len];
    let mut deg_out = vec![0.0f32; out_len];
    for (i, &s) in kept.iter().enumerate() {
        let o = i * HOP;
        for ((dst, &wk), &src) in clean_out[o..o + N_FRAME]
            .iter_mut()
            .zip(w)
            .zip(&clean[s..s + N_FRAME])
        {
            *dst += wk * src;
        }
        for ((dst, &wk), &src) in deg_out[o..o + N_FRAME]
            .iter_mut()
            .zip(w)
            .zip(&degraded[s..s + N_FRAME])
        {
            *dst += wk * src;
        }
    }
    (clean_out, deg_out)
}

/// Half-open FFT bin range `[lo, hi)` of one one-third-octave band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Band {
    lo: usize,
    hi: usize,
}

/// The 15 one-third-octave bands, as half-open FFT bin ranges.
///
/// Band `k` is centred at `MIN_FREQ · 2^(k/3)` with edges at
/// `MIN_FREQ · 2^((2k∓1)/6)` (the geometric means of adjacent centres); each
/// edge snaps to its nearest FFT bin. Consecutive bands share an edge frequency
/// by construction, so the ranges tile without gaps or overlap.
fn third_octave_bands() -> Vec<Band> {
    let bins = NFFT / 2 + 1;
    let bin_hz = f64::from(FS) / NFFT as f64;
    // `argmin_i (i·bin_hz − hz)²` is `round(hz / bin_hz)`. An exact .5 tie would
    // make the reference (numpy `argmin`, first minimum) round down where Rust's
    // `round` rounds up, but a tie needs `2^((2k±1)/6)` to be rational, and
    // `(2k±1)` is odd so it never is.
    let nearest = |hz: f64| ((hz / bin_hz).round() as usize).min(bins - 1);
    (0..NUM_BANDS)
        .map(|k| {
            let k = k as f64;
            Band {
                lo: nearest(MIN_FREQ * 2f64.powf((2.0 * k - 1.0) / 6.0)),
                hi: nearest(MIN_FREQ * 2f64.powf((2.0 * k + 1.0) / 6.0)),
            }
        })
        .collect()
}

/// Per-frame one-third-octave band envelopes of `sig`, returned as
/// `(frames, values)` with `values` row-major `[band, frame]`.
///
/// Each entry is `sqrt(Σ_bin |X|²)` over the band's bins of the 512-point real
/// FFT of the windowed, zero-padded 256-sample frame.
fn band_envelopes(sig: &[f32], w: &[f32], bands: &[Band], plan: &RealFftPlan) -> (usize, Vec<f64>) {
    let starts: Vec<usize> = (0..)
        .map(|i| i * HOP)
        .take_while(|s| s + N_FRAME < sig.len())
        .collect();
    let frames = starts.len();
    let mut out = vec![0.0f64; bands.len() * frames];
    let mut buf = vec![0.0f32; NFFT];
    for (f, &s) in starts.iter().enumerate() {
        // Windowed frame at the FRONT of the buffer, zeros after — the
        // reference's `rfft(frame * w, n=512)`.
        for (dst, (&v, &wk)) in buf.iter_mut().zip(sig[s..s + N_FRAME].iter().zip(w)) {
            *dst = v * wk;
        }
        buf[N_FRAME..].fill(0.0);
        let spec = plan.forward(&buf);
        for (b, band) in bands.iter().enumerate() {
            let power: f64 = spec[band.lo..band.hi]
                .iter()
                .map(|c| f64::from(c.re) * f64::from(c.re) + f64::from(c.im) * f64::from(c.im))
                .sum();
            out[b * frames + f] = power.sqrt();
        }
    }
    (frames, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic broadband pseudo-noise in [-1, 1) — no RNG dependency
    /// (NFR-DS-02). Broadband matters: a pure tone leaves most of the 15 bands
    /// empty, and an empty band contributes 0 to the mean rather than 1.
    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
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

    // 1.6 s at the analysis rate → ~123 frames → ~94 segments.
    const N: usize = 16_000;

    #[test]
    fn band_layout_matches_the_published_one_third_octave_grid() {
        let bands = third_octave_bands();
        assert_eq!(bands.len(), NUM_BANDS);

        // 10 kHz / 512 = 19.53125 Hz per bin.
        // Band 0:  150·2^(-1/6) = 133.63 Hz → bin 7 ;  150·2^(1/6) = 168.37 → bin 9.
        assert_eq!(bands[0], Band { lo: 7, hi: 9 });
        // Band 1:  168.37 → bin 9 ; 150·2^(1/2) = 212.13 → bin 11.
        assert_eq!(bands[1], Band { lo: 9, hi: 11 });
        // Band 14: 150·2^(9/2) = 3394.11 → bin 174 ; 150·2^(29/6) = 4276.31 → bin 219.
        assert_eq!(bands[14], Band { lo: 174, hi: 219 });

        for (k, b) in bands.iter().enumerate() {
            assert!(b.lo < b.hi, "band {k} is empty: {b:?}");
            assert!(b.hi <= NFFT / 2 + 1, "band {k} runs past the spectrum");
        }
        // Adjacent bands share an edge frequency, so the grid tiles exactly.
        for pair in bands.windows(2) {
            assert_eq!(pair[0].hi, pair[1].lo, "one-third-octave grid has a gap");
        }
    }

    #[test]
    fn identical_signals_score_one() {
        // With x == y the energy normalisation gives α ≈ 1, the clip never
        // binds (1 + 10^0.75 ≈ 6.62 > 1), and every correlation is 1.
        let x = noise(N, 0x1111_0001);
        let d = stoi(&x, &x, FS).unwrap();
        assert!(
            d > 0.999 && d <= 1.0 + 1e-9,
            "identical signals must score ~1.0 (got {d})"
        );
    }

    #[test]
    fn degradation_lowers_the_score_monotonically() {
        let x = noise(N, 0x2222_0002);
        let nz = noise(N, 0x3333_0003);
        let clean = stoi(&add(&x, &nz, 0.05), &x, FS).unwrap();
        let dirty = stoi(&add(&x, &nz, 1.0), &x, FS).unwrap();
        assert!(
            clean < 1.0,
            "any degradation must score below 1 (got {clean})"
        );
        assert!(
            clean > dirty,
            "more interference must score lower: {clean} vs {dirty}"
        );
    }

    #[test]
    fn uncorrelated_signal_scores_low() {
        let x = noise(N, 0x4444_0004);
        let junk = noise(N, 0x5555_0005);
        let d = stoi(&junk, &x, FS).unwrap();
        assert!(
            d < 0.6,
            "an unrelated signal must not look intelligible (got {d})"
        );
    }

    #[test]
    fn all_zero_input_scores_zero_rather_than_nan() {
        // Degenerate but reachable (a model that emitted silence). The epsilon
        // guards in the correlation must keep this finite.
        let zeros = vec![0.0f32; N];
        let d = stoi(&zeros, &zeros, FS).unwrap();
        assert!(d.is_finite(), "all-zero input must not produce NaN");
        assert!(d.abs() < 1e-9, "all-zero input should score ~0 (got {d})");
    }

    #[test]
    fn resampling_path_runs_and_preserves_a_perfect_score() {
        // 16 kHz input exercises vokra_ops::resample before the analysis.
        let x = noise(2 * N, 0x6666_0006);
        let d = stoi(&x, &x, 16_000).unwrap();
        assert!(
            d > 0.99,
            "resampling a signal against itself must stay ~1.0 (got {d})"
        );
    }

    #[test]
    fn too_short_input_errors_instead_of_returning_the_1e_5_sentinel() {
        // 2000 samples → ~14 frames < SEG_FRAMES.
        let x = noise(2_000, 0x7777_0007);
        let err = stoi(&x, &x, FS);
        assert!(err.is_err(), "fewer than 30 frames must be a loud error");
    }

    #[test]
    fn length_mismatch_and_empty_and_zero_rate_error() {
        let x = noise(N, 0x8888_0008);
        assert!(stoi(&x[..N - 1], &x, FS).is_err());
        let empty: [f32; 0] = [];
        assert!(stoi(&empty, &empty, FS).is_err());
        assert!(stoi(&x, &x, 0).is_err());
    }

    #[test]
    fn non_finite_samples_error_rather_than_scoring_nan() {
        let mut x = noise(N, 0x9999_0009);
        let reference = x.clone();
        x[100] = f32::NAN;
        assert!(stoi(&x, &reference, FS).is_err());
        x[100] = f32::INFINITY;
        assert!(stoi(&x, &reference, FS).is_err());
    }

    #[test]
    fn silence_removal_drops_quiet_frames() {
        // A loud half followed by a −60 dB half: the quiet frames are >40 dB
        // down and must be dropped, shortening the analysed signal.
        let loud = noise(N, 0xAAAA_000A);
        let mut sig = loud.clone();
        for v in &mut sig[N / 2..] {
            *v *= 0.001;
        }
        let full = window(Window::Hann, N_FRAME + 2, WindowSymmetry::Symmetric);
        let w: &[f32] = &full[1..=N_FRAME];
        let (kept, _) = remove_silent_frames(&sig, &sig, w);
        assert!(!kept.is_empty());
        // Roughly half the frames are >40 dB down, so the packed result must be
        // well under three quarters of the input. (A plain `< sig.len()` would
        // pass vacuously: overlap-add always returns slightly fewer samples than
        // it was given, even when nothing is dropped.)
        assert!(
            kept.len() < (3 * sig.len()) / 4,
            "quiet frames should have been removed ({} vs {})",
            kept.len(),
            sig.len()
        );

        // A uniformly loud signal keeps everything (nothing is 40 dB down).
        let (all, _) = remove_silent_frames(&loud, &loud, w);
        assert!(
            all.len() > kept.len(),
            "a uniformly loud signal must keep more than a half-silent one"
        );
    }

    #[test]
    fn deterministic() {
        let x = noise(N, 0xBBBB_000B);
        let y = add(&x, &noise(N, 0xCCCC_000C), 0.2);
        assert_eq!(stoi(&y, &x, FS).unwrap(), stoi(&y, &x, FS).unwrap());
    }

    #[test]
    fn metric_surface_is_wired() {
        let x = noise(N, 0xDDDD_000D);
        let y = add(&x, &noise(N, 0xEEEE_000E), 0.2);
        let m = Stoi::new();
        assert_eq!(m.name(), "stoi");
        assert_eq!(m.direction(), Direction::HigherIsBetter);
        assert_eq!(m.analysis_rate(), FS);
        assert_eq!(m.eval_audio(&y, &x, FS).unwrap(), stoi(&y, &x, FS).unwrap());
        assert_eq!(m.score(&y, &x, FS).unwrap(), stoi(&y, &x, FS).unwrap());
        // Vacuous while `Stoi` is a unit struct — pinned anyway so the day it
        // gains a configuration field, `Default` and `new()` diverging fails
        // here rather than shipping silently.
        #[allow(
            clippy::default_constructed_unit_structs,
            reason = "regression guard for the day this stops being a unit struct"
        )]
        {
            assert_eq!(Stoi::default(), Stoi::new());
        }
    }
}
