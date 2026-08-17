//! Scale-invariant SNR / SDR and plain SDR — the reference-based **separation
//! and enhancement** quality metrics (the scores every `sepformer` /
//! `conv_tasnet` / `demucs` / `gtcrn` / `facebook_denoiser` wave landed models
//! is reported against upstream).
//!
//! Three metrics share one projection core:
//!
//! ```text
//! α          = ⟨ŝ, s⟩ / ‖s‖²            (least-squares gain onto the reference)
//! s_target   = α · s
//! e_noise    = ŝ − s_target
//! SI-SNR(dB) = 10 · log10( ‖s_target‖² / ‖e_noise‖² )
//! SDR(dB)    = 10 · log10( ‖s‖²        / ‖ŝ − s‖²   )   (no α — not scale-invariant)
//! ```
//!
//! where `s` is the reference (clean/target) source and `ŝ` the estimate. The
//! scale-invariant forms are unchanged when the estimate is multiplied by any
//! non-zero constant — `α` absorbs the gain — which is the entire point: a
//! separation network that recovers the source up to an arbitrary output gain
//! is not penalised for it. Plain [`Sdr`] *does* move under a gain change, and
//! the unit tests assert exactly that contrast.
//!
//! # Primary sources
//!
//! - **SI-SNR** — Luo & Mesgarani, *"Conv-TasNet: Surpassing Ideal
//!   Time-Frequency Magnitude Masking for Speech Separation"*,
//!   <https://arxiv.org/abs/1809.07454>. Its training objective is the formula
//!   above, applied after both signals are normalised to zero mean; that mean
//!   removal is modelled here by [`MeanRemoval::ZeroMean`].
//! - **SI-SDR** — Le Roux, Wisdom, Erdogan & Hershey, *"SDR — half-baked or
//!   well done?"*, <https://arxiv.org/abs/1811.02508>.
//!
//! # Is SI-SDR different from SI-SNR?
//!
//! **No — under the common definition they are the same formula**, and this
//! module computes them with the same code path. The names differ because the
//! two papers arrived at the identical quantity from different directions
//! (a separation training loss vs. a critique of BSS-Eval's SDR); the
//! literature uses them interchangeably. The only variation worth encoding is
//! the *preprocessing*: Conv-TasNet zero-means both signals first, Le Roux et
//! al.'s definition does not. That is a one-line difference in preparation, not
//! a different metric, so it is exposed as [`MeanRemoval`] on both types rather
//! than being invented into a second formula. [`SiSnr`] defaults to
//! [`MeanRemoval::ZeroMean`] and [`SiSdr`] to [`MeanRemoval::AsIs`]; flip either
//! and the two produce **bit-identical** scores (asserted by
//! `si_snr_and_si_sdr_are_the_same_formula`).
//!
//! # Degenerate inputs are loud, never `inf` / `NaN` (FR-EX-08)
//!
//! A dB ratio is unbounded at both ends, and a metric runner that averages over
//! a corpus is destroyed by a single `±inf`. Every way of reaching an unbounded
//! or undefined score is therefore a [`VokraError::InvalidArgument`], not a
//! sentinel and not a silently-added epsilon:
//!
//! | condition                                    | why it is rejected                     |
//! |----------------------------------------------|----------------------------------------|
//! | `estimate.len() != reference.len()`          | silent truncation would score a lie    |
//! | empty input                                  | no samples to project                  |
//! | non-finite sample (`NaN` / `±inf`)           | would propagate to a `NaN` score       |
//! | zero-energy reference                        | `α` divides by `‖s‖² = 0`              |
//! | zero-energy estimate                         | almost always a model that emitted nothing |
//! | **zero-energy residual** (perfect estimate)  | ratio `→ +∞`                           |
//! | **zero-energy projection** (orthogonal)      | ratio `→ 0`, dB `→ −∞`                 |
//!
//! The "perfect estimate" row deserves emphasis because it is a *legitimate*
//! input: `si_snr_db(x, x)` — and, for the scale-invariant metrics, `si_snr_db(c·x, x)`
//! for any `c ≠ 0` — is an **error**, not a large number. The alternative
//! (clamping at some `MAX_DB`) was rejected because there is no principled value
//! to clamp at; any constant would be a fabricated number, which this repo does
//! not ship. A *near*-perfect estimate returns a large finite value as usual,
//! and `near_perfect_reconstruction_is_large_and_finite` pins that the metric
//! does not fall over as the boundary is approached.
//!
//! # Provenance
//!
//! This promotes the private `si_snr_db` helper inside
//! `crates/vokra-ops/tests/parity_denoise_dfn3.rs` (the DeepFilterNet3
//! real-weight parity oracle) to a first-class metric. That helper **stays where
//! it is** — the crate edge runs `vokra-eval → vokra-ops`, so a `vokra-ops` test
//! cannot call this module — so `matches_the_dfn3_parity_oracle_shape` pins the
//! two against each other instead, and a future refactor here cannot silently
//! change what the DFN3 parity leg measures. The accumulation shape is
//! deliberately identical to that helper — two passes in `f64`, `s_target`
//! re-summed rather than folded into the algebraically-equal closed form
//! `‖e‖² = ‖ŝ‖² − α⟨ŝ,s⟩` — because the closed form catastrophically cancels
//! for a good estimate, exactly where the metric matters most.

use super::{AudioRefMetric, Direction, Metric};
use vokra_core::{Result, VokraError};

/// Whether the scale-invariant projection removes each signal's DC mean first.
///
/// See the module docs: this is the *only* documented difference between the
/// SI-SNR and SI-SDR definitions, and it is preprocessing rather than a
/// different formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeanRemoval {
    /// Subtract each signal's own mean before projecting — the preprocessing
    /// Conv-TasNet (<https://arxiv.org/abs/1809.07454>) specifies for SI-SNR.
    ZeroMean,
    /// Project the signals as they are — Le Roux et al.
    /// (<https://arxiv.org/abs/1811.02508>) define SI-SDR without mean removal.
    AsIs,
}

/// Scale-invariant signal-to-noise ratio, in dB (Conv-TasNet's SI-SNR).
///
/// Defaults to [`MeanRemoval::ZeroMean`]. Higher is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SiSnr {
    mean_removal: MeanRemoval,
}

impl SiSnr {
    /// SI-SNR as Conv-TasNet defines it: zero-mean both signals, then project.
    pub fn new() -> Self {
        Self {
            mean_removal: MeanRemoval::ZeroMean,
        }
    }

    /// SI-SNR with an explicit [`MeanRemoval`] policy.
    ///
    /// `SiSnr::with_mean_removal(MeanRemoval::AsIs)` is bit-identical to
    /// [`SiSdr::new`].
    pub fn with_mean_removal(mean_removal: MeanRemoval) -> Self {
        Self { mean_removal }
    }

    /// The configured mean-removal policy.
    pub fn mean_removal(&self) -> MeanRemoval {
        self.mean_removal
    }

    /// Scores `estimate` against `reference` in dB.
    ///
    /// # Errors
    ///
    /// See the module docs' degenerate-input table — every unbounded or
    /// undefined case is a [`VokraError::InvalidArgument`].
    pub fn score(&self, estimate: &[f32], reference: &[f32]) -> Result<f64> {
        scale_invariant_db(estimate, reference, self.mean_removal, "si_snr")
    }
}

impl Default for SiSnr {
    fn default() -> Self {
        Self::new()
    }
}

/// Scale-invariant signal-to-distortion ratio, in dB (Le Roux et al.'s SI-SDR).
///
/// Defaults to [`MeanRemoval::AsIs`]. **The formula is the same as [`SiSnr`]'s**
/// — see the module docs. Higher is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SiSdr {
    mean_removal: MeanRemoval,
}

impl SiSdr {
    /// SI-SDR as Le Roux et al. define it: project the signals as they are.
    pub fn new() -> Self {
        Self {
            mean_removal: MeanRemoval::AsIs,
        }
    }

    /// SI-SDR with an explicit [`MeanRemoval`] policy.
    ///
    /// `SiSdr::with_mean_removal(MeanRemoval::ZeroMean)` is bit-identical to
    /// [`SiSnr::new`].
    pub fn with_mean_removal(mean_removal: MeanRemoval) -> Self {
        Self { mean_removal }
    }

    /// The configured mean-removal policy.
    pub fn mean_removal(&self) -> MeanRemoval {
        self.mean_removal
    }

    /// Scores `estimate` against `reference` in dB.
    ///
    /// # Errors
    ///
    /// See the module docs' degenerate-input table.
    pub fn score(&self, estimate: &[f32], reference: &[f32]) -> Result<f64> {
        scale_invariant_db(estimate, reference, self.mean_removal, "si_sdr")
    }
}

impl Default for SiSdr {
    fn default() -> Self {
        Self::new()
    }
}

/// Plain (**not** scale-invariant) signal-to-distortion ratio, in dB:
/// `10 · log10(‖s‖² / ‖ŝ − s‖²)`.
///
/// Unlike [`SiSnr`] / [`SiSdr`] this penalises an output-gain mismatch, which is
/// what you want when the absolute level is part of the contract (a denoiser or
/// an AEC that must not change loudness) and what you do *not* want when
/// scoring a separation network whose output gain is arbitrary. Higher is
/// better.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Sdr;

impl Sdr {
    /// Builds the metric (it carries no configuration).
    pub fn new() -> Self {
        Self
    }

    /// Scores `estimate` against `reference` in dB.
    ///
    /// # Errors
    ///
    /// See the module docs' degenerate-input table.
    pub fn score(&self, estimate: &[f32], reference: &[f32]) -> Result<f64> {
        sdr_db(estimate, reference)
    }
}

/// SI-SNR in dB with Conv-TasNet's zero-mean preprocessing.
///
/// # Errors
///
/// See the module docs' degenerate-input table.
pub fn si_snr_db(estimate: &[f32], reference: &[f32]) -> Result<f64> {
    scale_invariant_db(estimate, reference, MeanRemoval::ZeroMean, "si_snr")
}

/// SI-SDR in dB with Le Roux et al.'s definition (no mean removal).
///
/// Identical in formula to [`si_snr_db`]; see the module docs.
///
/// # Errors
///
/// See the module docs' degenerate-input table.
pub fn si_sdr_db(estimate: &[f32], reference: &[f32]) -> Result<f64> {
    scale_invariant_db(estimate, reference, MeanRemoval::AsIs, "si_sdr")
}

/// Plain SDR in dB: `10 · log10(‖s‖² / ‖ŝ − s‖²)`.
///
/// # Errors
///
/// See the module docs' degenerate-input table.
pub fn sdr_db(estimate: &[f32], reference: &[f32]) -> Result<f64> {
    check_pair(estimate, reference, "sdr")?;

    let mut ref_energy = 0.0f64;
    let mut est_energy = 0.0f64;
    let mut residual = 0.0f64;
    for (&e, &r) in estimate.iter().zip(reference) {
        let (e, r) = (f64::from(e), f64::from(r));
        est_energy += e * e;
        ref_energy += r * r;
        let d = e - r;
        residual += d * d;
    }

    check_finite(est_energy, ref_energy, "sdr")?;
    check_energies(est_energy, ref_energy, "sdr", MeanRemoval::AsIs)?;
    if residual <= 0.0 {
        return Err(perfect_estimate_error("sdr", false));
    }
    Ok(10.0 * (ref_energy / residual).log10())
}

/// The shared scale-invariant core behind [`si_snr_db`] / [`si_sdr_db`].
///
/// `who` names the caller in error messages so a runner can tell which metric
/// rejected the pair.
fn scale_invariant_db(
    estimate: &[f32],
    reference: &[f32],
    mean_removal: MeanRemoval,
    who: &str,
) -> Result<f64> {
    check_pair(estimate, reference, who)?;

    let n = estimate.len() as f64;
    let (est_mean, ref_mean) = match mean_removal {
        MeanRemoval::ZeroMean => (
            estimate.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
            reference.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
        ),
        MeanRemoval::AsIs => (0.0, 0.0),
    };

    // Pass 1: the inner product and both energies (post mean removal).
    let mut dot = 0.0f64;
    let mut ref_energy = 0.0f64;
    let mut est_energy = 0.0f64;
    for (&e, &r) in estimate.iter().zip(reference) {
        let e = f64::from(e) - est_mean;
        let r = f64::from(r) - ref_mean;
        dot += e * r;
        ref_energy += r * r;
        est_energy += e * e;
    }

    check_finite(est_energy, ref_energy, who)?;
    check_energies(est_energy, ref_energy, who, mean_removal)?;

    // Least-squares gain of the reference that best explains the estimate.
    let alpha = dot / ref_energy;

    // Pass 2: ‖s_target‖² and ‖e_noise‖² are re-summed rather than folded into
    // the algebraically-equal `‖e‖² = ‖ŝ‖² − α·⟨ŝ,s⟩`. That closed form loses
    // nearly every significant digit for a good estimate (the two terms agree
    // to many places), which is precisely the regime the metric exists to
    // resolve.
    let mut target = 0.0f64;
    let mut noise = 0.0f64;
    for (&e, &r) in estimate.iter().zip(reference) {
        let e = f64::from(e) - est_mean;
        let r = f64::from(r) - ref_mean;
        let s = alpha * r;
        target += s * s;
        let d = e - s;
        noise += d * d;
    }

    if noise <= 0.0 {
        return Err(perfect_estimate_error(who, true));
    }
    if target <= 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "{who}: the estimate is exactly orthogonal to the reference \
             (⟨estimate, reference⟩ = 0), so the projected target has zero \
             energy and the ratio is 0 — the dB score is unbounded below. \
             Refusing to return -inf; check that the estimate and the reference \
             are the same source and are time-aligned."
        )));
    }
    Ok(10.0 * (target / noise).log10())
}

/// Shape validation shared by every metric in this module.
fn check_pair(estimate: &[f32], reference: &[f32], who: &str) -> Result<()> {
    if estimate.len() != reference.len() {
        return Err(VokraError::InvalidArgument(format!(
            "{who}: length mismatch — estimate has {} samples, reference has {}. \
             Both must cover the same mono span at the same rate; this metric \
             does not truncate to the shorter buffer (a silent truncation would \
             score a different signal than the caller asked about).",
            estimate.len(),
            reference.len()
        )));
    }
    if estimate.is_empty() {
        return Err(VokraError::InvalidArgument(format!(
            "{who}: empty input — there are no samples to project"
        )));
    }
    Ok(())
}

/// Rejects `NaN` / `±inf` samples before they can reach `log10` and produce a
/// `NaN` score. Both energies are sums of squares, so a single non-finite
/// sample makes the corresponding energy non-finite — one cheap check covers
/// the whole buffer.
fn check_finite(est_energy: f64, ref_energy: f64, who: &str) -> Result<()> {
    if !est_energy.is_finite() || !ref_energy.is_finite() {
        return Err(VokraError::InvalidArgument(format!(
            "{who}: input contains a non-finite sample (NaN or ±inf) — \
             estimate energy {est_energy}, reference energy {ref_energy}. \
             Refusing to return a NaN score."
        )));
    }
    Ok(())
}

/// Rejects a zero-energy reference or estimate.
fn check_energies(
    est_energy: f64,
    ref_energy: f64,
    who: &str,
    mean_removal: MeanRemoval,
) -> Result<()> {
    let qualifier = match mean_removal {
        MeanRemoval::ZeroMean => {
            " (after mean removal — a constant/DC signal \
             becomes all-zero here)"
        }
        MeanRemoval::AsIs => "",
    };
    if ref_energy <= 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "{who}: the reference has zero energy{qualifier}, so the score is \
             undefined — ‖reference‖² is the scale-invariant projection's \
             divisor and plain SDR's numerator"
        )));
    }
    if est_energy <= 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "{who}: the estimate has zero energy{qualifier}. An all-zero \
             estimate is almost always a model that produced nothing, so it is \
             rejected rather than scored — a finite number here would look like \
             a poor-but-valid result."
        )));
    }
    Ok(())
}

/// The shared "the residual has no energy" error (see the module docs on why
/// this is an error rather than a clamped ceiling).
fn perfect_estimate_error(who: &str, scale_invariant: bool) -> VokraError {
    let extra = if scale_invariant {
        "the estimate is bit-identical to the reference, or an exact scalar \
         multiple of it (which a scale-invariant metric treats as perfect)"
    } else {
        "the estimate is bit-identical to the reference"
    };
    VokraError::InvalidArgument(format!(
        "{who}: {extra}, so the residual has zero energy and the ratio is \
         unbounded above. Refusing to return +inf, and refusing to clamp at an \
         invented ceiling; handle 'perfect reconstruction' explicitly at the \
         call site."
    ))
}

impl Metric for SiSnr {
    fn name(&self) -> &str {
        "si_snr"
    }
    fn direction(&self) -> Direction {
        Direction::HigherIsBetter
    }
}

impl Metric for SiSdr {
    fn name(&self) -> &str {
        "si_sdr"
    }
    fn direction(&self) -> Direction {
        Direction::HigherIsBetter
    }
}

impl Metric for Sdr {
    fn name(&self) -> &str {
        "sdr"
    }
    fn direction(&self) -> Direction {
        Direction::HigherIsBetter
    }
}

/// Rejects a nonsensical rate for the rate-agnostic time-domain metrics.
///
/// SI-SNR / SI-SDR / SDR are pure time-domain ratios — they have no front end,
/// so any rate is acceptable as long as *both* buffers are at it (which the
/// caller asserts by passing one rate for the pair). `0` is still refused: it
/// signals an uninitialised caller, and silently accepting it would hide the
/// bug behind a plausible-looking dB number.
fn check_rate(sample_rate: u32, who: &str) -> Result<()> {
    if sample_rate == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{who}: sample_rate must be non-zero (the metric itself is \
             rate-agnostic, but 0 indicates an uninitialised caller)"
        )));
    }
    Ok(())
}

impl AudioRefMetric for SiSnr {
    fn eval_audio(&self, hyp: &[f32], reference: &[f32], sample_rate: u32) -> Result<f64> {
        check_rate(sample_rate, "si_snr")?;
        self.score(hyp, reference)
    }
}

impl AudioRefMetric for SiSdr {
    fn eval_audio(&self, hyp: &[f32], reference: &[f32], sample_rate: u32) -> Result<f64> {
        check_rate(sample_rate, "si_sdr")?;
        self.score(hyp, reference)
    }
}

impl AudioRefMetric for Sdr {
    fn eval_audio(&self, hyp: &[f32], reference: &[f32], sample_rate: u32) -> Result<f64> {
        check_rate(sample_rate, "sdr")?;
        self.score(hyp, reference)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 16_000;

    fn tone(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SR as f32).sin())
            .collect()
    }

    /// Deterministic pseudo-noise in [-1, 1) — no RNG dependency (NFR-DS-02).
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

    fn scaled(a: &[f32], c: f32) -> Vec<f32> {
        a.iter().map(|x| c * x).collect()
    }

    fn offset(a: &[f32], dc: f32) -> Vec<f32> {
        a.iter().map(|x| x + dc).collect()
    }

    #[test]
    fn perfect_reconstruction_errors_loudly() {
        // The documented decision: a zero-energy residual is an error, not
        // +inf and not a clamped ceiling. See the module docs.
        let x = tone(440.0, 4_000);
        assert!(si_snr_db(&x, &x).is_err());
        assert!(si_sdr_db(&x, &x).is_err());
        assert!(sdr_db(&x, &x).is_err());

        // For the SCALE-INVARIANT metrics an exact scalar multiple is equally
        // "perfect" — α absorbs the gain, the residual is zero.
        let doubled = scaled(&x, 2.0);
        assert!(si_snr_db(&doubled, &x).is_err());
        assert!(si_sdr_db(&doubled, &x).is_err());
        // …but plain SDR scores it finitely: the gain IS the distortion.
        let plain = sdr_db(&doubled, &x).unwrap();
        assert!(
            plain.is_finite(),
            "plain SDR must score a gain change finitely (got {plain})"
        );
    }

    #[test]
    fn near_perfect_reconstruction_is_large_and_finite() {
        // Approaching the boundary must not blow up: a 1e-6 perturbation of a
        // unit-amplitude tone is ~30x above the f32 quantum, so the score is
        // dominated by the intended perturbation and lands very high.
        let x = tone(440.0, 4_000);
        let almost = add(&x, &noise(x.len(), 0x1234_5678), 1e-6);
        let db = si_snr_db(&almost, &x).unwrap();
        assert!(
            db.is_finite() && db > 80.0 && db < 250.0,
            "near-perfect estimate should score very high but finite (got {db})"
        );
    }

    #[test]
    fn scale_invariance_holds_for_si_metrics_but_not_for_sdr() {
        // The whole point of the "SI" prefix.
        let x = tone(440.0, 4_000);
        let est = add(&x, &noise(x.len(), 0xACE1_0001), 0.1);

        // Tolerance derived, not tuned to make the test pass.
        //
        // Scaling by a power of two IS bit-exact (the exponent moves, the
        // mantissa does not), so c = 2.0 / 0.25 agree to the last bit. c = 3.7
        // does not: each f32 sample is re-rounded after the multiply, giving a
        // per-sample relative error of up to eps_f32 / 2 ~= 6e-8. The metric
        // forms a ratio of sums of squares, so that propagates to ~1.2e-7
        // relative on the ratio, and the dB conversion scales it by
        // d(10 log10 r)/d(ln r) = 10 / ln 10 ~= 4.34:
        //
        //     4.34 * 1.2e-7 ~= 5.2e-7 dB   (analytic worst case)
        //
        // Measured on this fixture at c = 3.7: 7.1e-8 dB. The bound below is
        // ~2x the analytic worst case, which still fails loudly on any real
        // scale dependence (a genuine one moves the score by whole dB, as the
        // plain-SDR leg of this same test demonstrates).
        const SI_INVARIANCE_TOL_DB: f64 = 1e-6;

        for c in [2.0f32, 3.7, 0.25] {
            let est_c = scaled(&est, c);

            let a = si_snr_db(&est, &x).unwrap();
            let b = si_snr_db(&est_c, &x).unwrap();
            assert!(
                (a - b).abs() < SI_INVARIANCE_TOL_DB,
                "SI-SNR must be invariant under a x{c} gain: {a} vs {b}"
            );

            let a = si_sdr_db(&est, &x).unwrap();
            let b = si_sdr_db(&est_c, &x).unwrap();
            assert!(
                (a - b).abs() < SI_INVARIANCE_TOL_DB,
                "SI-SDR must be invariant under a x{c} gain: {a} vs {b}"
            );

            // Plain SDR is NOT: a gain change is a real distortion for it.
            let a = sdr_db(&est, &x).unwrap();
            let b = sdr_db(&est_c, &x).unwrap();
            assert!(
                (a - b).abs() > 1.0,
                "plain SDR must move under a x{c} gain: {a} vs {b}"
            );
        }
    }

    #[test]
    fn si_snr_and_si_sdr_are_the_same_formula() {
        // The honest claim from the module docs, pinned: the two names are one
        // formula, and only the mean-removal preprocessing differs. Match the
        // preprocessing and the scores are bit-identical.
        let x = tone(440.0, 4_000);
        let est = add(&x, &noise(x.len(), 0x5EED_0002), 0.2);

        let a = SiSnr::with_mean_removal(MeanRemoval::AsIs)
            .score(&est, &x)
            .unwrap();
        let b = SiSdr::new().score(&est, &x).unwrap();
        assert_eq!(a, b, "SI-SNR(AsIs) must equal SI-SDR(AsIs) bit-for-bit");

        let a = SiSnr::new().score(&est, &x).unwrap();
        let b = SiSdr::with_mean_removal(MeanRemoval::ZeroMean)
            .score(&est, &x)
            .unwrap();
        assert_eq!(a, b, "SI-SNR(ZeroMean) must equal SI-SDR(ZeroMean)");
    }

    #[test]
    fn mean_removal_is_what_separates_the_two_defaults() {
        // A DC offset on the estimate is invisible to the zero-mean form and
        // visible to the as-is form — the one documented behavioural difference.
        // (Signal energy is kept high so the f32 rounding of `+ 0.25` stays far
        // below the assertion tolerance.)
        let x = tone(440.0, 4_000);
        let est = add(&x, &noise(x.len(), 0xBEEF_0003), 0.1);
        let est_dc = offset(&est, 0.25);

        let a = SiSnr::new().score(&est, &x).unwrap();
        let b = SiSnr::new().score(&est_dc, &x).unwrap();
        assert!(
            (a - b).abs() < 0.05,
            "zero-mean SI-SNR should ignore a DC offset: {a} vs {b}"
        );

        let a = SiSdr::new().score(&est, &x).unwrap();
        let b = SiSdr::new().score(&est_dc, &x).unwrap();
        assert!(
            (a - b).abs() > 1.0,
            "as-is SI-SDR should see a DC offset: {a} vs {b}"
        );
    }

    #[test]
    fn uncorrelated_noise_scores_low() {
        let x = tone(440.0, 4_000);
        let junk = noise(x.len(), 0xF00D_0004);
        let db = si_snr_db(&junk, &x).unwrap();
        assert!(
            db < 0.0,
            "an estimate uncorrelated with the reference must score below 0 dB (got {db})"
        );
        assert!(sdr_db(&junk, &x).unwrap() < 5.0);
    }

    #[test]
    fn more_noise_scores_worse() {
        let x = tone(440.0, 4_000);
        let nz = noise(x.len(), 0x0A0A_0005);
        let clean = si_snr_db(&add(&x, &nz, 0.01), &x).unwrap();
        let dirty = si_snr_db(&add(&x, &nz, 0.5), &x).unwrap();
        assert!(
            clean > dirty,
            "less noise must score higher: {clean} vs {dirty}"
        );
    }

    #[test]
    fn zero_energy_inputs_error_loudly() {
        let x = tone(440.0, 1_000);
        let zeros = vec![0.0f32; x.len()];
        // Zero reference.
        assert!(si_snr_db(&x, &zeros).is_err());
        assert!(si_sdr_db(&x, &zeros).is_err());
        assert!(sdr_db(&x, &zeros).is_err());
        // Zero estimate.
        assert!(si_snr_db(&zeros, &x).is_err());
        assert!(si_sdr_db(&zeros, &x).is_err());
        assert!(sdr_db(&zeros, &x).is_err());
    }

    #[test]
    fn constant_signal_is_zero_energy_after_mean_removal() {
        // DC-only input survives the as-is form but is all-zero once the mean
        // is removed — the qualifier in the error message exists for this case.
        let dc = vec![0.5f32; 1_000];
        let x = tone(440.0, 1_000);
        assert!(si_snr_db(&x, &dc).is_err(), "zero-mean form must reject DC");
        assert!(
            si_sdr_db(&x, &dc).is_ok(),
            "as-is form has a non-zero reference energy for DC"
        );
    }

    #[test]
    fn orthogonal_estimate_errors_rather_than_returning_neg_inf() {
        // ⟨[1, -1], [1, 1]⟩ = 0 exactly in IEEE-754, so the projection has zero
        // energy and the dB score would be -inf.
        let reference = [1.0f32, 1.0];
        let estimate = [1.0f32, -1.0];
        assert!(si_sdr_db(&estimate, &reference).is_err());
        // Plain SDR has no projection, so it scores this pair finitely.
        assert!(sdr_db(&estimate, &reference).unwrap().is_finite());
    }

    #[test]
    fn length_mismatch_errors_rather_than_truncating() {
        let a = tone(440.0, 1_000);
        let b = tone(440.0, 900);
        assert!(si_snr_db(&a, &b).is_err());
        assert!(si_sdr_db(&a, &b).is_err());
        assert!(sdr_db(&a, &b).is_err());
    }

    #[test]
    fn empty_input_errors() {
        let empty: [f32; 0] = [];
        assert!(si_snr_db(&empty, &empty).is_err());
        assert!(sdr_db(&empty, &empty).is_err());
    }

    #[test]
    fn non_finite_samples_error_rather_than_scoring_nan() {
        let mut x = tone(440.0, 1_000);
        let reference = x.clone();
        x[10] = f32::NAN;
        assert!(si_snr_db(&x, &reference).is_err());
        assert!(sdr_db(&x, &reference).is_err());

        let mut y = tone(440.0, 1_000);
        y[10] = f32::INFINITY;
        assert!(si_snr_db(&y, &reference).is_err());
        assert!(sdr_db(&y, &reference).is_err());
    }

    #[test]
    fn deterministic() {
        let x = tone(440.0, 4_000);
        let est = add(&x, &noise(x.len(), 0xD00D_0006), 0.1);
        assert_eq!(si_snr_db(&est, &x).unwrap(), si_snr_db(&est, &x).unwrap());
        assert_eq!(si_sdr_db(&est, &x).unwrap(), si_sdr_db(&est, &x).unwrap());
        assert_eq!(sdr_db(&est, &x).unwrap(), sdr_db(&est, &x).unwrap());
    }

    #[test]
    fn metric_surface_is_wired() {
        let x = tone(440.0, 4_000);
        let est = add(&x, &noise(x.len(), 0xC0DE_0007), 0.1);

        let m = SiSnr::new();
        assert_eq!(m.name(), "si_snr");
        assert_eq!(m.direction(), Direction::HigherIsBetter);
        assert_eq!(m.mean_removal(), MeanRemoval::ZeroMean);
        assert_eq!(
            m.eval_audio(&est, &x, SR).unwrap(),
            si_snr_db(&est, &x).unwrap()
        );

        let m = SiSdr::new();
        assert_eq!(m.name(), "si_sdr");
        assert_eq!(m.direction(), Direction::HigherIsBetter);
        assert_eq!(m.mean_removal(), MeanRemoval::AsIs);
        assert_eq!(
            m.eval_audio(&est, &x, SR).unwrap(),
            si_sdr_db(&est, &x).unwrap()
        );

        let m = Sdr::new();
        assert_eq!(m.name(), "sdr");
        assert_eq!(m.direction(), Direction::HigherIsBetter);
        assert_eq!(
            m.eval_audio(&est, &x, SR).unwrap(),
            sdr_db(&est, &x).unwrap()
        );

        // Defaults agree with the constructors. Vacuous while these are unit
        // structs — which is exactly why it is worth pinning: the day one of
        // them gains a configuration field (a `MeanRemoval` default, say),
        // `Default` and `new()` can silently disagree, and this line starts
        // failing instead of the divergence shipping.
        #[allow(
            clippy::default_constructed_unit_structs,
            reason = "regression guard for the day these stop being unit structs"
        )]
        {
            assert_eq!(SiSnr::default(), SiSnr::new());
            assert_eq!(SiSdr::default(), SiSdr::new());
            assert_eq!(Sdr::default(), Sdr::new());
        }
    }

    #[test]
    fn zero_sample_rate_is_rejected_by_the_trait_surface() {
        let x = tone(440.0, 1_000);
        let est = add(&x, &noise(x.len(), 0x9999_0008), 0.1);
        assert!(SiSnr::new().eval_audio(&est, &x, 0).is_err());
        assert!(SiSdr::new().eval_audio(&est, &x, 0).is_err());
        assert!(Sdr::new().eval_audio(&est, &x, 0).is_err());
    }

    #[test]
    fn matches_the_dfn3_parity_oracle_shape() {
        // The private helper this promotes lives in
        // crates/vokra-ops/tests/parity_denoise_dfn3.rs. Recompute it inline and
        // require agreement, so a future refactor of the metric cannot silently
        // change what the DeepFilterNet3 real-weight parity leg measures.
        fn oracle(est: &[f32], reference: &[f32]) -> f64 {
            let n = est.len() as f64;
            let me: f64 = est.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
            let mr: f64 = reference.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
            let mut dot = 0.0f64;
            let mut rr = 0.0f64;
            for (&e, &r) in est.iter().zip(reference) {
                let (e, r) = (f64::from(e) - me, f64::from(r) - mr);
                dot += e * r;
                rr += r * r;
            }
            let alpha = dot / rr;
            let mut sig = 0.0f64;
            let mut err = 0.0f64;
            for (&e, &r) in est.iter().zip(reference) {
                let (e, r) = (f64::from(e) - me, f64::from(r) - mr);
                let s = alpha * r;
                sig += s * s;
                let d = e - s;
                err += d * d;
            }
            10.0 * (sig / err).log10()
        }

        let x = tone(440.0, 4_000);
        let est = add(&x, &noise(x.len(), 0x7777_0009), 0.2);
        assert_eq!(si_snr_db(&est, &x).unwrap(), oracle(&est, &x));
    }
}
