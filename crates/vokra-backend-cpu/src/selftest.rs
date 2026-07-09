//! Single-binary runtime self-consistency proof (M1-05).
//!
//! [`selftest`] is a small, allocation-light routine that a *shipped* Vokra
//! binary can call to prove — on the actual host CPU, at run time — that the
//! ISA path it selected ([`crate::active_isa`]) computes the same results as
//! the portable scalar oracle. It underpins the "single binary runs on
//! x86-64 **and** ARM64 via runtime dispatch" completion claim (FR-BE-01,
//! FR-EX-06): the one artifact detects its host features, dispatches to the
//! matching SIMD kernels, and here re-checks that dispatch against the scalar
//! reference before doing real work.
//!
//! It is deliberately an **internal oracle** (differential-vs-scalar), never a
//! fabricated reference number: every host-supported SIMD path is run against
//! the scalar path on the same fixed, seeded inputs and the maximum absolute
//! deviation is asserted under [`SELFTEST_ATOL`] (mirroring the ceiling used
//! by `tests/differential.rs`, itself well under the FP32 parity bound
//! NFR-QL-01 `atol = 0.01`). A deviation above the tolerance means a SIMD
//! kernel is miscompiled or the CPU is misdetected on this host, so the
//! backend is not trustworthy here — reported as
//! [`VokraError::BackendUnavailable`].
//!
//! Intended callers: the `vokra-cli` `doctor` / `bench` surface (M1-10) and
//! the ASR demo's one-line ISA log (M0-06-T26) — both can print
//! [`SelftestReport`] to show which path is live and that it is self-consistent.

use vokra_core::{Result, VokraError};

use crate::dispatch::{self, active_isa};
use crate::features::{CpuFeatures, IsaPath};
use crate::kernels;

/// Absolute-tolerance ceiling for the self-consistency check.
///
/// Matches the loosest per-kernel tolerance in `tests/differential.rs`
/// (`GEMM_ATOL`, whose error grows with the K-reduction length) and stays far
/// under the FP32 parity bound NFR-QL-01 `atol = 0.01`. A companion relative
/// term ([`SELFTEST_RTOL`]) absorbs GEMM's magnitude-dependent rounding.
pub const SELFTEST_ATOL: f32 = 1e-3;

/// Relative-tolerance companion to [`SELFTEST_ATOL`] (GEMM reduction rounding).
pub const SELFTEST_RTOL: f32 = 1e-4;

/// Outcome of [`selftest`]: which path is live and how far the host SIMD
/// kernels drifted from the scalar oracle.
#[derive(Debug, Clone)]
pub struct SelftestReport {
    /// The ISA path this process selected (host default, or the
    /// `VOKRA_CPU_ISA` override) — see [`crate::active_isa`].
    pub active_isa: IsaPath,
    /// The CPU features detected on this host.
    pub features: CpuFeatures,
    /// The SIMD paths that were cross-checked against the scalar oracle.
    ///
    /// Empty on a host with no SIMD path (only scalar is available), in which
    /// case the check is a trivial identity and always passes.
    pub checked_paths: Vec<IsaPath>,
    /// The largest absolute deviation observed between any host SIMD kernel
    /// and the scalar oracle (0.0 when no SIMD path was checked).
    pub max_abs_diff: f32,
    /// The absolute tolerance applied ([`SELFTEST_ATOL`]).
    pub tolerance: f32,
}

impl core::fmt::Display for SelftestReport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cpu selftest OK: active={} features(avx2={} fma={} neon={} rvv_v={} zvfh={}) checked={:?} max_abs_diff={:.3e} (atol {:.0e})",
            self.active_isa,
            self.features.avx2,
            self.features.fma,
            self.features.neon,
            self.features.rvv_v,
            self.features.rvv_zvfh,
            self.checked_paths,
            self.max_abs_diff,
            self.tolerance,
        )
    }
}

/// Minimal reproducible PRNG (xorshift64*), so the selftest needs no external
/// `rand` dependency (NFR-DS-02 zero-dependency invariant) and produces the
/// same inputs on every host.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let bits = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32; // 24 bits
        bits as f32 / (1u32 << 24) as f32 * 2.0 - 1.0 // [-1, 1)
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Compares `simd` against the scalar `oracle`, updating `max_abs_diff` and
/// returning an explicit error the moment any element exceeds the
/// atol+rtol band — a genuine self-consistency failure (never a silent pass).
fn compare(
    kernel: &str,
    isa: IsaPath,
    oracle: &[f32],
    simd: &[f32],
    max_abs_diff: &mut f32,
) -> Result<()> {
    for (i, (&o, &s)) in oracle.iter().zip(simd).enumerate() {
        let diff = (o - s).abs();
        if diff > *max_abs_diff {
            *max_abs_diff = diff;
        }
        let tol = SELFTEST_ATOL + SELFTEST_RTOL * o.abs();
        // `diff > tol` alone would let a NaN (from a corrupt SIMD result)
        // slip through, since every NaN comparison is false; fail on it too.
        if diff > tol || diff.is_nan() {
            return Err(VokraError::BackendUnavailable(format!(
                "cpu selftest failed: {isa} kernel `{kernel}` deviates from the scalar oracle at \
                 index {i} (scalar={o}, {isa}={s}, |diff|={diff} > tol {tol}); the SIMD path is \
                 miscompiled or the host CPU is misdetected"
            )));
        }
    }
    Ok(())
}

/// Runs every host-supported kernel path against the scalar oracle on fixed
/// seeded inputs and returns a [`SelftestReport`], or
/// [`VokraError::BackendUnavailable`] if any path disagrees.
///
/// This is the runtime companion to the compile-time `tests/differential.rs`
/// harness: the same differential-vs-scalar idea, but callable from the
/// shipped binary so a deployed host can self-verify its dispatch
/// (FR-BE-01, FR-EX-06). It is cheap (small fixed shapes) and deterministic.
///
/// # Errors
/// Returns [`VokraError::BackendUnavailable`] if a host SIMD kernel's output
/// drifts from the scalar oracle beyond [`SELFTEST_ATOL`] (a miscompiled SIMD
/// path or a misdetected CPU), and propagates any
/// [`VokraError::InvalidArgument`] from the kernels (not expected here, since
/// all shapes are internally consistent).
pub fn selftest() -> Result<SelftestReport> {
    let features = CpuFeatures::detect();
    let active_isa = active_isa();

    // The SIMD paths this host can actually run (Scalar is the oracle, not a
    // "checked" path). At most one of Avx2 / Neon / Rvv is supported on any
    // given host (they are arch-exclusive). RVV is added in M3-13.
    let checked_paths: Vec<IsaPath> = [IsaPath::Avx2, IsaPath::Neon, IsaPath::Rvv]
        .into_iter()
        .filter(|&isa| features.supports(isa))
        .collect();

    let mut max_abs_diff = 0.0f32;

    // Fixed seeds per kernel keep the inputs identical across hosts and runs.
    let mut rng = Rng::new(0x5E1F_7E57);

    // GEMM 6x13x9 with bias — n = 13 exercises both the AVX2 (8) and NEON (4)
    // lane tails; the K = 9 reduction is where FP32 error is largest.
    let (m, n, k) = (6usize, 13usize, 9usize);
    let ga = rng.vec(m * k);
    let gb = rng.vec(k * n);
    let gbias = rng.vec(n);
    // Element-wise / activation inputs — len 41 = 5*8+1 = 10*4+1 (ragged tail
    // on both ISAs).
    let ex = rng.vec(41);
    let ey = rng.vec(41);
    // Softmax 3x17 and layer-norm 2x13 (reduction kernels with tails).
    let sm = rng.vec(3 * 17);
    let ln = rng.vec(2 * 13);
    let gamma = rng.vec(13);
    let beta = rng.vec(13);
    // GEMV 7x19 with per-row bias — k = 19 spans the 16-lane (NEON) /
    // single-8-wide + scalar (AVX2) reduction remainders of the logits-head
    // fast path. (Generated last so the other kernels' seeded inputs are
    // unchanged.)
    let (gvm, gvk) = (7usize, 19usize);
    let gva = rng.vec(gvm * gvk);
    let gvx = rng.vec(gvk);
    let gvbias = rng.vec(gvm);

    // Scalar oracles (computed once).
    let mut o_gemm = vec![0.0; m * n];
    kernels::gemm_f32_on(
        IsaPath::Scalar,
        m,
        n,
        k,
        &ga,
        &gb,
        Some(&gbias),
        &mut o_gemm,
    )?;
    let mut o_gemv = vec![0.0; gvm];
    kernels::gemv_f32_on(
        IsaPath::Scalar,
        gvm,
        gvk,
        &gva,
        &gvx,
        Some(&gvbias),
        &mut o_gemv,
    )?;
    let mut o_add = vec![0.0; ex.len()];
    kernels::add_f32_on(IsaPath::Scalar, &ex, &ey, &mut o_add)?;
    let mut o_mul = vec![0.0; ex.len()];
    kernels::mul_f32_on(IsaPath::Scalar, &ex, &ey, &mut o_mul)?;
    let mut o_relu = vec![0.0; ex.len()];
    kernels::relu_f32_on(IsaPath::Scalar, &ex, &mut o_relu)?;
    let mut o_sig = vec![0.0; ex.len()];
    kernels::sigmoid_f32_on(IsaPath::Scalar, &ex, &mut o_sig)?;
    let mut o_tanh = vec![0.0; ex.len()];
    kernels::tanh_f32_on(IsaPath::Scalar, &ex, &mut o_tanh)?;
    let mut o_gelu = vec![0.0; ex.len()];
    kernels::gelu_f32_on(IsaPath::Scalar, &ex, &mut o_gelu)?;
    let mut o_sm = vec![0.0; sm.len()];
    kernels::softmax_f32_on(IsaPath::Scalar, &sm, &mut o_sm, 3, 17)?;
    let mut o_ln = vec![0.0; ln.len()];
    let eps = kernels::LAYER_NORM_DEFAULT_EPS;
    kernels::layer_norm_f32_on(IsaPath::Scalar, &ln, &mut o_ln, 2, 13, &gamma, &beta, eps)?;

    for &isa in &checked_paths {
        let mut buf = vec![0.0; m * n];
        kernels::gemm_f32_on(isa, m, n, k, &ga, &gb, Some(&gbias), &mut buf)?;
        compare("gemm", isa, &o_gemm, &buf, &mut max_abs_diff)?;

        let mut buf = vec![0.0; gvm];
        kernels::gemv_f32_on(isa, gvm, gvk, &gva, &gvx, Some(&gvbias), &mut buf)?;
        compare("gemv", isa, &o_gemv, &buf, &mut max_abs_diff)?;

        let mut buf = vec![0.0; ex.len()];
        kernels::add_f32_on(isa, &ex, &ey, &mut buf)?;
        compare("add", isa, &o_add, &buf, &mut max_abs_diff)?;
        kernels::mul_f32_on(isa, &ex, &ey, &mut buf)?;
        compare("mul", isa, &o_mul, &buf, &mut max_abs_diff)?;
        kernels::relu_f32_on(isa, &ex, &mut buf)?;
        compare("relu", isa, &o_relu, &buf, &mut max_abs_diff)?;
        kernels::sigmoid_f32_on(isa, &ex, &mut buf)?;
        compare("sigmoid", isa, &o_sig, &buf, &mut max_abs_diff)?;
        kernels::tanh_f32_on(isa, &ex, &mut buf)?;
        compare("tanh", isa, &o_tanh, &buf, &mut max_abs_diff)?;
        kernels::gelu_f32_on(isa, &ex, &mut buf)?;
        compare("gelu", isa, &o_gelu, &buf, &mut max_abs_diff)?;

        let mut buf = vec![0.0; sm.len()];
        kernels::softmax_f32_on(isa, &sm, &mut buf, 3, 17)?;
        compare("softmax", isa, &o_sm, &buf, &mut max_abs_diff)?;

        let mut buf = vec![0.0; ln.len()];
        kernels::layer_norm_f32_on(isa, &ln, &mut buf, 2, 13, &gamma, &beta, eps)?;
        compare("layer_norm", isa, &o_ln, &buf, &mut max_abs_diff)?;
    }

    // M2-04-T06: cross-check the fused log-mel dispatch path against the
    // scalar oracle on fixed seeded inputs. Runs the same differential
    // pattern as the other kernels — the failure surface is
    // `VokraError::BackendUnavailable` with a diagnostic pointing at the
    // fused_logmel kernel, so a shipped binary can report SIMD miscompile /
    // CPU misdetection in the log-mel front-end at run time.
    fused_logmel_selftest(&features, &checked_paths, &mut max_abs_diff)?;

    Ok(SelftestReport {
        active_isa,
        features,
        checked_paths,
        max_abs_diff,
        tolerance: SELFTEST_ATOL,
    })
}

/// M2-04-T06: cross-check the fused log-mel per-frame kernel across every
/// host-supported ISA path against the scalar oracle. Called from
/// [`selftest`] at feature-detect init and updates `max_abs_diff` with the
/// worst deviation observed. On disagreement, returns
/// [`VokraError::BackendUnavailable`] naming the fused_logmel kernel.
///
/// # Tolerance
///
/// The fused kernel's SIMD paths use polynomial `log10` approximations with
/// worst-case per-element error ≪ 1e-6, and the surrounding mel-band
/// accumulation is FMA-associated (bit-close to the scalar reduction). Uses
/// the [`SELFTEST_ATOL`] + [`SELFTEST_RTOL`] band, matching the other
/// kernels — still an order of magnitude inside the FP32 parity ceiling
/// NFR-QL-01 `atol = 0.01`.
fn fused_logmel_selftest(
    features: &CpuFeatures,
    checked_paths: &[IsaPath],
    max_abs_diff: &mut f32,
) -> Result<()> {
    // A Whisper-shaped tile so both the AVX2 eight-lane FMA tail
    // (`n_bins % 8 = 1`) and the NEON four-lane FMA tail (`n_bins % 4 = 1`)
    // exercise the ragged-tail scalar cleanup that must match the scalar
    // reference bit-close.
    let (n_mels, n_bins) = (16usize, 41usize);
    let mut rng = Rng::new(0x10E1_5EED);
    let mut weights = Vec::with_capacity(n_mels * n_bins);
    for _ in 0..n_mels * n_bins {
        // Filterbank weights are non-negative — draw in [0, 1] to match the
        // real Mel filter magnitudes; using signed inputs here would let
        // opposite-sign cancellations dominate the FP32 error budget.
        weights.push((rng.next_f32() + 1.0) * 0.5);
    }
    let mut power = Vec::with_capacity(n_bins);
    for _ in 0..n_bins {
        // Power spectrum entries are non-negative; magnitude range spans the
        // Whisper log-mel domain (small-signal floor to peak power).
        power.push(((rng.next_f32() + 1.0) * 0.5) * 1e3);
    }

    // Scalar oracle via the safe crate-public dispatch on `Scalar`.
    let mut oracle = vec![0.0f32; n_mels];
    let scalar_table = dispatch::table_for(IsaPath::Scalar)?;
    (scalar_table.fused_logmel)(&weights, &power, n_mels, n_bins, 1e-10, &mut oracle);

    for &isa in checked_paths {
        // `table_for` on an unsupported path is an explicit error (already
        // covered elsewhere); the filter here mirrors the invariant that
        // `checked_paths` only contains ISAs the host reports as supported.
        if !features.supports(isa) {
            continue;
        }
        let table = dispatch::table_for(isa)?;
        let mut buf = vec![0.0f32; n_mels];
        (table.fused_logmel)(&weights, &power, n_mels, n_bins, 1e-10, &mut buf);
        compare("fused_logmel", isa, &oracle, &buf, max_abs_diff)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selftest_passes_on_this_host() {
        let report = selftest().expect("cpu selftest must pass on the CI/dev host");
        // The active path must be one this host actually supports.
        assert!(report.features.supports(report.active_isa));
        // Every checked path is a genuine SIMD path the host supports, and the
        // active default path is included whenever it is not scalar.
        for &isa in &report.checked_paths {
            assert!(report.features.supports(isa));
            assert_ne!(isa, IsaPath::Scalar);
        }
        if report.active_isa != IsaPath::Scalar {
            assert!(report.checked_paths.contains(&report.active_isa));
        }
        // Self-consistency held within tolerance.
        assert!(
            report.max_abs_diff <= SELFTEST_ATOL,
            "max_abs_diff {} exceeded atol {SELFTEST_ATOL}",
            report.max_abs_diff
        );
    }

    #[test]
    fn report_display_is_one_line() {
        let s = selftest().unwrap().to_string();
        assert!(s.starts_with("cpu selftest OK:"));
        assert!(!s.contains('\n'));
    }
}
